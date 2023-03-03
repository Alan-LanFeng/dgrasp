from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import get_obj_pcd,get_args,repeat_label
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch

import joblib

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_ppo():
    actor = ppo_module.Actor(
        ppo_module.MLP(cfg['architecture']['policy_net'], nn.Tanh, nn.LeakyReLU, ob_dim, act_dim, False),
        ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0), device)

    critic = ppo_module.Critic(
        ppo_module.MLP(cfg['architecture']['value_net'], nn.Tanh, nn.LeakyReLU, ob_dim, 1, False), device)

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
torch.set_default_dtype(torch.double)
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
exp_path = home_path

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")


### load config
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))

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
mesh_path = "/local/home/lafeng/Desktop/raisim/raisim_grasp/rsc/meshes_simplified/008_pudding_box/mesh_aligned.obj"
obj_pcd = get_obj_pcd(mesh_path)
obj_pcd = np.repeat(obj_pcd[np.newaxis, ...], num_envs, 0)

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'],label=repeated_label,obj_pcd=obj_pcd)


### Setting dimensions from environments
ob_dim = env.num_obs-3
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
for update in range(args.num_iterations):
    start = time.time()

    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    ### Store policy
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': ppo.actor.architecture.state_dict(),
            'actor_distribution_state_dict': ppo.actor.distribution.state_dict(),
            'critic_architecture_state_dict': ppo.critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        env.save_scaling(saver.data_dir, str(update))

    next_obs,info = env.reset()
    print(info['table_pos'][0])
    for step in range(n_steps):
        #obs = env.observe(get_obj_pcd=False).astype('float64')
        obs = next_obs
        action = ppo.observe(obs)
        next_obs,reward, dones,_ = env.step(action.astype('float64'))
        reward.clip(min=reward_clip)

        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)
    obs,_ = env.observe()

    ### Update policy
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    ppo.actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()
    ### Log results
    mean_file_name = saver.data_dir + "/rewards.txt"
    np.savetxt(mean_file_name, avg_rewards)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(ppo.actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

