from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from raisimGymTorch.algo.ppo.pointnet_encoder import PointNetEncoder_mtr

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


class MLP_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_network, self).__init__()
        self.activation_fn = nn.LeakyReLU

        shape = [256, 128]

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


class pn_pcd(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        hidden_dim = 256
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

        self.obj_pcd_embed = PointNetEncoder_mtr(in_channels=3,hidden_dim=hidden_dim,out_channels=hidden_dim)

    def forward(self,obs):
        n_env,_ = obs.shape
        obj_pcd = obs[:,280:].reshape(n_env,-1,3)
        hand_info = obs[:,:280]

        #obj_pcd = obj_pcd.permute(0,2,1)
        obj_pcd_encode = self.obj_pcd_embed(obj_pcd)*0.2

        feature_inp = torch.cat([hand_info,obj_pcd_encode],dim=-1)
        pred = self.mlp(feature_inp)
        return pred

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_p=0.3):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(self._get_mlp_layer(input_size, hidden_sizes[0]))
        for h in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(self._get_mlp_layer(hidden_sizes[h], hidden_sizes[h + 1]))
        self.dropout = nn.Dropout(dropout_p)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def _get_mlp_layer(self, input_size, output_size):
        return nn.Sequential(
            nn.Conv1d(input_size, output_size, 1),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        for hidden in self.hidden_layers:
            x = hidden(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, dropout_p=0.0):
        super(PointNetEncoder, self).__init__()
        self.mlp = MLP(input_size=3, output_size=256, hidden_sizes=[64, 128, 256], dropout_p=dropout_p)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 256)
        return x


class PointNetDecoder(nn.Module):
    def __init__(self):
        super(PointNetDecoder, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024*3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 1024, 3)

class PointNetAutoEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.decoder = PointNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch['obj_pc']
        x = x.transpose(2, 1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x= x.transpose(2, 1)
        loss = self.chamfer_distance(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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


if __name__ == '__main__':
    points = torch.randn(2, 3, 778)
    print(points.size())
    pointnet = PointNetEncoder()
    print('params {}M'.format(sum(p.numel() for p in pointnet.parameters()) / 1000000.0))
    glb_feature, trans, trans_feature = pointnet(points)
    print(glb_feature.size())
