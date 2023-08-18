from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp_drop as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import get_obj_pcd,get_args,repeat_label,setup_seed,concat_dict
from raisimGymTorch.env.bin.dgrasp_drop import NormalSampler
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.dense_mlp as dense_mlp
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import wandb
import math

import joblib


def get_ppo(mod):
    actor = ppo_module.Actor(mod(ob_dim, act_dim),
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0,
                                                                               NormalSampler(act_dim)), device)

    critic = ppo_module.Critic(mod(ob_dim, 1), device)

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
if cfg['module'] == 'MLP':
    mod = ppo_module.MLP_network
    mod = dense_mlp.DeepMLP_network
    cfg['environment']['get_pcd'] = False
    cfg['environment']['extra_dim'] = 1
elif cfg['module'] == 'mcg':
    mod = ppo_module.pn_pcd
    cfg['environment']['get_pcd'] = True
    cfg['environment']['extra_dim'] = 601
if args.exp_name != 'test':
    wandb.init(project='dgrasp',config=cfg,name = args.exp_name)

cfg['seed']=args.seed

### get experiment parameters
pre_grasp_steps = cfg['environment']['pre_grasp_steps']
trail_steps = cfg['environment']['trail_steps']
reward_clip = cfg['environment']['reward_clip']


dict_labels=joblib.load("raisimGymTorch/data/baseline.pkl")

for key in dict_labels:
    #if key!=1:continue
    for key2 in dict_labels[key]:
        dict_labels[key][key2] = dict_labels[key][key2][:20]

if args.all_objects:
    dict_labels = concat_dict(dict_labels)
    repeated_label = repeat_label(dict_labels, args.num_repeats)
else:
    repeated_label = repeat_label(dict_labels[args.obj_id], args.num_repeats)
# for k,v in repeated_label.items():
#     repeated_label[k] = v[:200]

num_envs = repeated_label['final_qpos'].shape[0]
cfg['environment']['num_envs'] = num_envs
cfg["testing"] = True if args.test else False
print('num envs', num_envs)


env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'],label=repeated_label)

### Setting dimensions from environments
ob_dim = env.obsdim_for_agent
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps
total_steps = n_steps * env.num_envs

### Set up logging
saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + args.exp_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=False)


ppo = get_ppo(mod)

avg_rewards = []

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
            f'python raisimGymTorch/env/envs/dgrasp_test/runner.py -e {exp} -w {weight} -sd {sd} -ao')

    next_obs,info = env.reset()
    done_array = np.zeros(num_envs)
    reward_array = np.zeros([n_steps, num_envs])
    reward_info = info['reward_info']
    for step in range(n_steps):

        obs = next_obs
        #obs = pe.add(obs,step)
        action = ppo.act(obs)
        next_obs,reward, dones,info_ = env.step(action.astype('float32'))
        rw = info_['reward_info']
        for i in range(len(reward_info)):
            r_i = reward_info[i]
            for k in r_i.keys():
                r_i[k] = r_i[k] + rw[i][k]
        done_array+=dones
        reward.clip(min=reward_clip)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        reward_array[step] = reward
        #reward_ll_sum = reward_ll_sum + np.sum(reward)

    obs = next_obs
    ### Update policy
    dones = done_array.astype(bool)
    reward_array[:,dones] = 0
    reward_ll_sum = np.sum(reward_array)
    valid_step = total_steps - np.sum(dones)*n_steps
    success_rate = (num_envs - done_array.astype(bool).sum())/num_envs

    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / valid_step

    average_dones = done_sum / valid_step
    avg_rewards.append(average_ll_performance)

    ppo.actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.2).to(device))

    end = time.time()
    ### Log results
    mean_file_name = saver.data_dir + "/rewards.txt"
    np.savetxt(mean_file_name, avg_rewards)

    # average all rewards in reward_info by len(reward_info)
    # first sum over all envs then average over all steps
    for i in range(len(reward_info)):
        r_i = reward_info[i]
        for k in r_i.keys():
            r_i[k] = r_i[k] / n_steps
    r_0 = reward_info[0]
    for k in r_0.keys():
        for i in range(1,len(reward_info)):
            r_0[k] = r_0[k] + reward_info[i][k]
        r_0[k] = r_0[k] / len(reward_info)
    r_0['crush_rate'] = success_rate
    r_0['rewards'] = average_ll_performance

    wandb.log(r_0)

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

