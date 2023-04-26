import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from raisimGymTorch.algo.ppo.module import MLP_3, CG_stacked
import pytorch_lightning as pl
from manopth.manolayer import ManoLayer
import mano

class mcg_graspgen(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.save_hyperparameters()
        modality = cfg['modality']
        layer_num = cfg['layer_num']
        self.hidden_dim = cfg['hidden_dim']
        self.cfg = cfg
        self.CG_obj = CG_stacked(layer_num, self.hidden_dim)
        self.CG_pred = CG_stacked(layer_num, self.hidden_dim)
        self.obj_encode = MLP_3([3, 64, 128, self.hidden_dim])

        # self.apply(self._init_weights)
        self.anchor_embedding = nn.Embedding(modality, self.hidden_dim)
        self.vec_emb = MLP_3([6, 64, 128, self.hidden_dim])

        self.pred_head = MLP_3([self.hidden_dim*2, 256, 128, 48+3])
        self.prob_head = MLP_3([self.hidden_dim * 2, 256, 128, 1])

        self.MSE = torch.nn.MSELoss(reduction='none')
        self.CLS = torch.nn.CrossEntropyLoss()

        self.mano_layer = ManoLayer(mano_root='raisimGymTorch/data', use_pca=True, ncomps=45)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.5,verbose=True)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):

        hand_param_pred,prob = self.forward(batch)

        loss,loss_dict = self.compute_loss(hand_param_pred,prob,batch)
        # add a prefix to all the keys in loss_dict, prefix is the name of the stage
        loss_dict = {'train/'+k:v for k,v in loss_dict.items()}
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):

        hand_param_pred,prob = self.forward(batch)

        loss,loss_dict = self.compute_loss(hand_param_pred,prob,batch)

        loss_dict = {'val/'+k:v for k,v in loss_dict.items()}
        self.log_dict(loss_dict)

    def compute_loss(self,recon_param,prob,batch):
        bs = recon_param.shape[0]

        recon_param = recon_param.reshape(-1, 51)

        verts, recon_joints = self.mano_layer(th_pose_coeffs=recon_param[:, :48], th_trans=recon_param[:, 48:])

        recon_joints = recon_joints.reshape(bs, -1, *recon_joints.shape[-2:])/1000
        gt_joints = batch['hand_joints'].unsqueeze(1).repeat(1,6,1,1)

        dist = self.MSE(recon_joints, gt_joints).mean(-1).sum(-1)

        min_index = torch.argmin(dist, dim=-1)
        cls_loss = self.CLS(prob, min_index)
        vertices_loss = torch.gather(dist, dim=1, index=min_index.unsqueeze(-1)).mean()

        recon_param = recon_param.reshape(bs, -1, 51)
        gt_param = torch.cat([batch['hand_rot'],batch['hand_pca'],batch['hand_pos']],dim=-1)
        param_loss = self.MSE(recon_param, gt_param.unsqueeze(1)).mean(-1)
        param_loss = torch.gather(param_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

        loss = 100*vertices_loss + param_loss + cls_loss
        loss_dict = {}
        loss_dict['vert_loss'] = vertices_loss.item()
        loss_dict['param_loss'] = param_loss.item()
        loss_dict['cls_loss'] = cls_loss.item()
        loss_dict['loss'] = loss.item()

        return loss,loss_dict


    def forward(self, batch):

        wrist_pos = batch['hand_pos']#+offset

        norm_vec = wrist_pos/torch.norm(wrist_pos,dim=-1).unsqueeze(-1)

       # add rotation
        wrist_rot = batch['hand_rot']
        condition_vec = torch.cat([wrist_rot,norm_vec],dim=-1)

        vec_embedding = self.vec_emb(condition_vec)

        pc_embedding = self.obj_encode(batch['obj_pc'])
        b,n,d = pc_embedding.shape
        device = pc_embedding.device
        mask = torch.ones([b, n], device=device)

        pc_embedding, context_obj = self.CG_obj(pc_embedding, vec_embedding, mask)
        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)

        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_pred(anchors, context_obj, mask)
        context_obj = context_obj.unsqueeze(1).repeat(1,6,1)
        pred_embed = torch.cat([pred_embed,context_obj],dim=-1)
        output = self.pred_head(pred_embed)
        prob = self.prob_head(pred_embed)

        return output,prob.squeeze(-1)

class mcg_graspgen_uncon(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.save_hyperparameters()
        self.modality = cfg['modality']
        layer_num = cfg['layer_num']
        self.hidden_dim = cfg['hidden_dim']
        self.cfg = cfg
        self.CG_obj = CG_stacked(layer_num, self.hidden_dim)
        self.CG_pred = CG_stacked(layer_num, self.hidden_dim)
        self.pc_encode = MLP_3([3, 256, 512, self.hidden_dim])
        # self.apply(self._init_weights)
        self.anchor_embedding = nn.Embedding(self.modality, self.hidden_dim)

        self.pred_head = MLP_3([self.hidden_dim*2, 256, 128, 51])
        self.prob_head = MLP_3([self.hidden_dim * 2, 256, 128, 1])

        self.MSE = torch.nn.MSELoss(reduction='none')
        self.CLS = torch.nn.CrossEntropyLoss()

        self.mano_layer = ManoLayer(mano_root='networks', use_pca=True, ncomps=45)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[75], gamma=0.1,verbose=True)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):

        hand_param_pred,prob = self.forward(batch)

        loss,loss_dict = self.compute_loss(hand_param_pred,prob,batch)

        self.log("train", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        hand_param_pred,prob = self.forward(batch)

        loss,loss_dict = self.compute_loss(hand_param_pred,prob,batch)

        self.log("val", loss,sync_dist=True)

        return loss

    def compute_loss(self,recon_param,prob,batch):
        bs = recon_param.shape[0]

        recon_param = recon_param.reshape(-1, 51)

        verts, recon_joints = self.mano_layer(th_pose_coeffs=recon_param[:, :48], th_trans=recon_param[:, 48:])

        recon_joints = recon_joints.reshape(bs, -1, *recon_joints.shape[-2:])/1000
        gt_joints = batch['hand_joints'].unsqueeze(1).repeat(1,self.modality,1,1)

        dist = self.MSE(recon_joints, gt_joints).mean(-1).sum(-1)

        min_index = torch.argmin(dist, dim=-1)
        cls_loss = self.CLS(prob, min_index)
        vertices_loss = torch.gather(dist, dim=1, index=min_index.unsqueeze(-1)).mean()

        recon_param = recon_param.reshape(bs, -1, 51)
        gt_param = torch.cat([batch['hand_rot'],batch['hand_pca'],batch['hand_pos']],dim=-1)
        param_loss = self.MSE(recon_param, gt_param.unsqueeze(1)).mean(-1)
        param_loss = torch.gather(param_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

        loss = 100*vertices_loss + param_loss + cls_loss
        loss_dict = {}
        loss_dict['vert_loss'] = vertices_loss.item()
        loss_dict['param_loss'] = param_loss.item()
        loss_dict['cls_loss'] = cls_loss.item()

        return loss,loss_dict


    def forward(self, batch):

        pc_embedding = self.pc_encode(batch['obj_pc'])
        b,n,d = pc_embedding.shape
        device = pc_embedding.device
        mask = torch.ones([b, n], device=device)
        init_context = torch.ones([b, d], device=device)

        pc_embedding, context_obj = self.CG_obj(pc_embedding, init_context, mask)

        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_pred(anchors, context_obj, mask)
        context_obj = context_obj.unsqueeze(1).repeat(1,self.modality,1)
        pred_embed = torch.cat([pred_embed,context_obj],dim=-1)
        output = self.pred_head(pred_embed)
        prob = self.prob_head(pred_embed)

        return output,prob.squeeze(-1)





