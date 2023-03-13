import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


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
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

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


class MLP_pcd(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        hidden_dim = 128
        self.hidden_dim = hidden_dim
        self.CG_1d = CG_stacked(3, hidden_dim)
        self.CG_pcd = CG_stacked(3, hidden_dim)
        shape = [128, 64]

        modules = [nn.Linear(hidden_dim, shape[0]), nn.LeakyReLU()]
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
        self.hand_embed = nn.Linear(121,hidden_dim)
        self.label_embed = nn.Linear(143,hidden_dim)
        self.obj_embed = nn.Linear(16,hidden_dim)
        self.obj_pcd_embed = nn.Linear(3,hidden_dim)

    def forward(self,obs):
        n_env,_ = obs.shape
        device = obs.device
        obj_pcd = obs[:,280:].reshape(n_env,-1,3)
        hand_info = obs[:,:121]
        label_info = obs[:,121:264]
        obj_info = obs[:,264:280]
        hand_encode = self.hand_embed(hand_info).unsqueeze(1)
        label_encode = self.label_embed(label_info).unsqueeze(1)
        obj_info_encode = self.obj_embed(obj_info).unsqueeze(1)
        meta_encode = torch.cat([hand_encode,label_encode,obj_info_encode],dim=1)

        context = torch.ones([n_env,self.hidden_dim]).to(device)
        obj_pcd_encode = self.obj_pcd_embed(obj_pcd)
        pcd_num = obj_pcd.shape[1]
        mask = torch.ones([n_env,3]).to(device)
        meta_encode, context_meta = self.CG_1d(meta_encode,context,mask)
        mask = torch.ones([n_env, pcd_num]).to(device)
        obj_pcd_encode, context_obj = self.CG_pcd(obj_pcd_encode, context_meta, mask)

        pred = self.mlp(context_obj)
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
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3])
        )
    def forward(self, x):
        x = self.mlp(x)
        return x

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

class MCG_block(nn.Module):
    def __init__(self,hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, inp, context,mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp=inp * context
        inp = inp.masked_fill(mask==0,torch.tensor(-1e9))
        context = torch.max(inp,dim=1)[0]
        return inp,context


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
