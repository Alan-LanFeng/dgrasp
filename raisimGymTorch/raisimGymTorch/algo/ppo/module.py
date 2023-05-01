from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from raisimGymTorch.helper.utils import get_inv
from scipy.spatial.transform import Rotation as R
from raisimGymTorch.helper.utils import dgrasp_to_mano,show_pointcloud_objhand

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture(obs).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = nn.LeakyReLU
        shape = [128, 128]
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]
        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))

        self.mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.mlp, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]
    def forward(self,obs):
        return self.mlp(obs)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class mcg_encoder(nn.Module):

    def __init__(self,mcg_graspgen):
        super().__init__()
        self.hidden_dim = 128
        layer_num = 3
        self.vec_emb = MLP_3([6, 64, 128, self.hidden_dim])
        self.obj_encode = MLP_3([3, 64, 128, self.hidden_dim])
        self.CG_obj = CG_stacked(layer_num, self.hidden_dim)
        # copy the param in mcg_graspgen to the encoder
        self.vec_emb.load_state_dict(mcg_graspgen.vec_emb.state_dict())
        self.obj_encode.load_state_dict(mcg_graspgen.obj_encode.state_dict())
        self.CG_obj.load_state_dict(mcg_graspgen.CG_obj.state_dict())


    def forward(self,vec,obj):

        vec_embedding = self.vec_emb(vec)

        pc_embedding = self.obj_encode(obj)

        b,n,d = pc_embedding.shape
        device = pc_embedding.device
        mask = torch.ones([b, n], device=device)
        _, context_obj = self.CG_obj(pc_embedding, vec_embedding, mask)

        return context_obj

class mcg_pretrain(nn.Module):
    def __init__(self,input_size, output_size,mcg_graspgen):
        super(mcg_pretrain, self).__init__()
        # self.mcg_encoder = mcg_encoder(mcg_graspgen)
        # # freeze the encoder
        # for param in self.mcg_encoder.parameters():
        #     param.requires_grad = False

        self.input_shape = [input_size]
        self.output_shape = [output_size]

        self.fc_obj = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU())

        self.fc_hand = nn.Sequential(nn.Linear(280, 128), nn.LeakyReLU())
        self.output = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

    def forward(self,obs):
        n_env, _ = obs.shape
        device = obs.device
        #obj_pcd = obs[:, 280:].reshape(n_env, -1, 3)
        obs_hand = obs[:, :280]
        obj_pcd_encode = obs[:, 280:]

    # # Get object position and rotation is the wrist frame
    #     obj_pos = -obs_hand[:, -16:-13]
    #     obj_euler = obs_hand[:, -13:-10]
    #     # Get wrist position and rotation in the object frame
    #     hand_pos,hand_rot = get_inv(obj_pos.cpu().detach().numpy(),obj_euler.cpu().detach().numpy())
    #     hand_pos = torch.tensor(hand_pos,dtype=torch.float32,device=device)
    #
    #     # idx = 0
    #     # gc = obs_hand[idx, :51]
    #     # gc[:3] = hand_pos
    #     # gc[3:6] = torch.tensor(hand_rot,dtype=torch.float32,device=device)
    #     # verts, joints = dgrasp_to_mano(gc.cpu().detach().numpy())
    #     # show_pointcloud_objhand(verts, obj_pcd[idx].reshape(-1, 3).detach().cpu().numpy())
    #
    #     # change hand_rot to rotation vector
    #     hand_rot = R.from_euler('XYZ', hand_rot, degrees=True).as_rotvec()
    #     hand_rot = torch.tensor(hand_rot,dtype=torch.float32,device=device)
    #
    #     # normalize the hand position to unit vector
    #     hand_pos = hand_pos / torch.norm(hand_pos, dim=1, keepdim=True)
    #
    #     vec = torch.cat([hand_pos, hand_rot], dim=1)
    #     obj_pcd_encode = self.mcg_encoder(vec, obj_pcd)
    #     self.obj_pcd_encode = self.fc_obj(obj_pcd_encode)

        hand_encode = self.fc_hand(obs_hand)


        output = self.output(torch.cat([hand_encode, obj_pcd_encode], dim=1))

        return output


class mcg_pcd(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        hidden_dim = 128
        self.hidden_dim = hidden_dim
        #self.CG_1d = CG_stacked(3, hidden_dim)
        self.CG_pcd = CG_stacked(2, hidden_dim)

        shape = [256, 128]

        modules = [nn.Linear(hidden_dim*2, shape[0]), nn.LeakyReLU()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(nn.LeakyReLU())
            scale.append(np.sqrt(2))
        modules.append(nn.Linear(shape[-1], output_size))

        self.mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))
        self.init_weights(self.mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

        # self.hand_embed = MLP_3([121,128,128,hidden_dim])
        # self.label_embed = MLP_3([143, 128, 128, hidden_dim])
        # self.obj_embed = MLP_3([16, 128, 128, hidden_dim])
        # self.obj_pcd_embed = MLP_3([3, 128, 128, hidden_dim])
        self.hand_embed = MLP_2([280,128,hidden_dim])
        # self.label_embed = nn.Linear(143,hidden_dim)
        # self.obj_embed = nn.Linear(16,hidden_dim)
        self.obj_pcd_embed = nn.Linear(3,hidden_dim)

    def forward(self,obs):
        n_env,_ = obs.shape
        device = obs.device
        obj_pcd = obs[:,280:].reshape(n_env,-1,3)
        hand_info = obs[:,:280]

        hand_encode = self.hand_embed(hand_info)
        # label_info = obs[:,121:264]
        # obj_info = obs[:,264:280]
        # hand_encode = self.hand_embed(hand_info).unsqueeze(1)
        # label_encode = self.label_embed(label_info).unsqueeze(1)
        # obj_info_encode = self.obj_embed(obj_info).unsqueeze(1)
        # meta_encode = torch.cat([hand_encode,label_encode,obj_info_encode],dim=1)

        #context = torch.ones([n_env,self.hidden_dim]).to(device)
        obj_pcd_encode = self.obj_pcd_embed(obj_pcd)
        pcd_num = obj_pcd.shape[1]
        mask = torch.ones([n_env,pcd_num]).to(device)
        meta_encode, context_meta = self.CG_pcd(obj_pcd_encode,hand_encode,mask)

        context = torch.cat([hand_encode,context_meta],dim=-1)

        pred = self.mlp(context)
        return pred

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class pn_pcd(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        hidden_dim = 128
        self.hidden_dim = hidden_dim

        shape = [256, 128]
        modules = [nn.Linear(hidden_dim+280, shape[0]), nn.LeakyReLU()]
        scale = [np.sqrt(2)]
        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(nn.LeakyReLU())
            scale.append(np.sqrt(2))
        modules.append(nn.Linear(shape[-1], output_size))

        self.mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))
        self.init_weights(self.mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

        self.obj_pcd_embed = PointNetEncoder(self.hidden_dim)

    def forward(self,obs):
        n_env,_ = obs.shape
        obj_pcd = obs[:,280:].reshape(n_env,-1,3)
        hand_info = obs[:,:280]

        obj_pcd_encode = self.obj_pcd_embed(obj_pcd)

        feature_inp = torch.cat([hand_info,obj_pcd_encode],dim=-1)
        pred = self.mlp(feature_inp)
        return pred

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class mlp_pcd(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        hidden_dim = 128
        self.hidden_dim = hidden_dim

        shape = [256, 128]
        modules = [nn.Linear(hidden_dim+280, shape[0]), nn.LeakyReLU()]
        scale = [np.sqrt(2)]
        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(nn.LeakyReLU())
            scale.append(np.sqrt(2))
        modules.append(nn.Linear(shape[-1], output_size))

        self.mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))
        self.init_weights(self.mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

        self.obj_pcd_embed = PointNetEncoder(self.hidden_dim)

    def forward(self,obs):
        n_env,_ = obs.shape
        obj_pcd = obs[:,280:].reshape(n_env,-1,3)
        hand_info = obs[:,:280]

        obj_pcd_encode = self.obj_pcd_embed(obj_pcd)

        feature_inp = torch.cat([hand_info,obj_pcd_encode],dim=-1)
        pred = self.mlp(feature_inp)
        return pred

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class MLP_3(nn.Module):
    def __init__(self, dims):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LayerNorm(dims[1]),
            nn.GELU(),
            nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]),
            nn.GELU(),
            nn.Linear(dims[2], dims[3])
        )
    def forward(self, x):
        x = self.mlp(x)
        return x

class MLP_2(nn.Module):
    def __init__(self, dims):
        super(MLP_2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LayerNorm(dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )
    def forward(self, x):
        x = self.mlp(x)
        return x

class MCG_block(nn.Module):
    def __init__(self,hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, inp, context,mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp=inp * context
        inp = inp.masked_fill(mask==0,torch.tensor(-1e9))
        context = torch.max(inp,dim=1)[0]
        return inp,context

class CG_stacked(nn.Module):
    def __init__(self,stack_num,hidden_dim):
        super(CG_stacked, self).__init__()
        self.CGs = nn.ModuleList()
        self.stack_num = stack_num
        for i in range(stack_num):
            self.CGs.append(MCG_block(hidden_dim))

    def forward(self, inp, context, mask):

        inp_,context_ = self.CGs[0](inp,context,mask)
        for i in range(1,self.stack_num):
            inp,context = self.CGs[i](inp_,context_,mask)
            inp_ = (inp_*i+inp)/(i+1)
            context_ = (context_*i+context)/(i+1)
        return inp_,context_



class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std


class PointNetEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, hidden_dim, 1)

        self.linear = nn.Linear(hidden_dim,hidden_dim)
        torch.nn.init.orthogonal_(self.linear.weight, gain=np.sqrt(2))
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x = x.transpose(2,1)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, self.hidden_dim)
        # x = self.layer_norm(self.linear(x))
        x = x.transpose(2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hidden_dim)
        return x

if __name__ == '__main__':
    points = torch.randn(2, 3, 778)
    print(points.size())
    pointnet = PointNetEncoder()
    print('params {}M'.format(sum(p.numel() for p in pointnet.parameters()) / 1000000.0))
    glb_feature, trans, trans_feature = pointnet(points)
    print(glb_feature.size())
