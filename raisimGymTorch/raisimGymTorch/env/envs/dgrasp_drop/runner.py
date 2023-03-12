from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp_drop as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import get_obj_pcd,get_args,repeat_label,setup_seed
from raisimGymTorch.env.bin.dgrasp_drop import NormalSampler
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import wandb
import math

import joblib

class PE():
    """
    Implement the PE function.
    """
    def __init__(self, d_model, max_len=5000):
        # Compute the positional encodings once in log space.
        pe = np.zeros([max_len, d_model])
        position = np.arange(0, max_len)[:,np.newaxis].astype(np.float32)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)[:,:-1]
        self.pe = pe.astype(np.float32)
    def add(self,obs,step):
        return obs+ self.pe[[step]]


def get_ppo():
    actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0,
                                                                               NormalSampler(act_dim)), device)

    critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1), device)

    ppo = PPO.PPO(actor=actor,
                  critic=critic,
                  num_envs=num_envs,
                  num_transitions_per_env=n_steps,
                  num_learning_epochs=4,
                  gamma=0.996,
                  lam=0.95,
                  num_mini_batches=4,
                  device=device,
                  log_dir=saver.data_dir,
                  shuffle_batch=False
                  )
    return ppo


### configuration of command line arguments

args = get_args()
setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
exp_path = home_path

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")


### load config
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))
wandb.init(project='dgrasp',config=cfg)

### set seed
if args.seed != 1:
    cfg['seed']=args.seed

### get experiment parameters
num_envs = cfg['environment']['num_envs']
pre_grasp_steps = cfg['environment']['pre_grasp_steps']
trail_steps = cfg['environment']['trail_steps']
reward_clip = cfg['environment']['reward_clip']


dict_labels=joblib.load("raisimGymTorch/data/dexycb_train_labels.pkl")
repeated_label = repeat_label(dict_labels[args.obj_id],args.num_repeats)


num_envs = repeated_label['final_qpos'].shape[0]
cfg['environment']['num_envs'] = num_envs
cfg["testing"] = True if args.test else False
print('num envs', num_envs)

# get obj pcd
mesh_path = "../rsc/meshes_simplified/008_pudding_box/mesh_aligned.obj"
obj_pcd = get_obj_pcd(mesh_path)
obj_pcd = np.repeat(obj_pcd[np.newaxis, ...], num_envs, 0)

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'],label=repeated_label,obj_pcd=obj_pcd)

### Setting dimensions from environments
ob_dim = env.num_obs+1
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * env.num_envs

### Set up logging
saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + args.exp_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=False)

ppo = get_ppo()

avg_rewards = []
eval_interval = 200
#pe = PE(ob_dim,n_steps)

for update in range(args.num_iterations):
    start = time.time()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    ### Store policy
    if update % cfg['environment']['eval_every_n'] == 0 and update:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': ppo.actor.architecture.state_dict(),
            'actor_distribution_state_dict': ppo.actor.distribution.state_dict(),
            'critic_architecture_state_dict': ppo.critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        env.save_scaling(saver.data_dir, str(update))
        pathes = saver.data_dir.split('/')
        sd = pathes[-3]
        exp = pathes[-2]
        weight = pathes[-1]+ "/full_" + str(update) + '.pt'
        os.system(
            f'python raisimGymTorch/env/envs/dgrasp_test/runner.py -o  7 -e {exp} -w {weight} -sd {sd} -t')

    next_obs,info = env.reset()
    done_array = np.zeros(num_envs)
    for step in range(n_steps):

        obs = next_obs
        step_obs = np.zeros([num_envs,1]).astype('float32')
        step_obs[:] = step/n_steps
        obs = np.concatenate([obs,step_obs],axis=1)
        #obs = pe.add(obs,step)
        action = ppo.act(obs)
        next_obs,reward, dones,_ = env.step(action.astype('float32'))
        done_array+=dones
        reward.clip(min=reward_clip)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)
    obs,_ = env.observe()

    ### Update policy
    success_rate = (num_envs - done_array.astype(bool).sum())/num_envs

    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    ppo.actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()
    ### Log results
    mean_file_name = saver.data_dir + "/rewards.txt"
    np.savetxt(mean_file_name, avg_rewards)

    results = {}
    results['rewards'] = average_ll_performance
    results['success_rate'] = success_rate
    wandb.log(results)

    # print('----------------------------------------------------')
    # print('{:>6}th iteration'.format(update))
    # print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    # print('{:<40} {:>6}'.format("success_rate: ", '{:0.6f}'.format(success_rate)))
    # print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    # print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    # print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
    #                                                                    * cfg['environment']['control_dt'])))
    # print('std: ')
    # print(np.exp(ppo.actor.distribution.std.cpu().detach().numpy()))
    # print('----------------------------------------------------\n')

