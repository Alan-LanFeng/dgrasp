import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from raisimGymTorch.helper.utils import first_nonzero

class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape).to(self.device)
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape).to(self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)
        self.mask = torch.ones(num_transitions_per_env, num_envs, 1).bool().to(self.device)
        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, actor_obs, critic_obs, actions, rewards, dones, values, actions_log_prob):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step].copy_(torch.from_numpy(critic_obs).to(self.device))
        self.actor_obs[self.step].copy_(torch.from_numpy(actor_obs).to(self.device))
        self.actions[self.step].copy_(actions.to(self.device))
        self.rewards[self.step].copy_(torch.from_numpy(rewards).view(-1, 1).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).view(-1, 1).to(self.device))
        self.values[self.step].copy_(values.to(self.device))
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1).to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        # self.dones[-3,0]=1
        # self.dones[-2,0]=1
        # self.values[:]=0.5
        # self.rewards[:]=1
        # last_values[:] = 0.5
        val_with_last = torch.cat([self.values,last_values.unsqueeze(0)],dim=0)
        a = first_nonzero(self.dones[...,0],0)
        for i in range(len(a)):
            if a[i]==-1:continue
            indx = a[i].item()
            val_with_last[indx+1:,i]=0
            self.rewards[indx+1:,i]=0
            self.mask[indx+1:,i]=False
            self.dones[indx:, i] = 1

        for step in reversed(range(self.num_transitions_per_env)):

            next_values = val_with_last[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - val_with_last[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + val_with_last[step]
            # a = self.returns.cpu().numpy()[...,0]
            # b = self.rewards.cpu().numpy()[..., 0]
            # c = self.values.cpu().numpy()[..., 0]
            # d = self.dones.cpu().numpy()[..., 0]
            # print()
        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices]
            critic_obs_batch = self.critic_obs.view(-1, *self.critic_obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            mask_batch = self.dones
            yield actor_obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_obs.view(-1, *self.critic_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size],\
                self.mask.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
