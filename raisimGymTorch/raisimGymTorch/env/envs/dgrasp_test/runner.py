from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dgrasp_test as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.dgrasp_test import NormalSampler
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.helper.utils import concat_dict
import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import joblib
from raisimGymTorch.helper.utils import get_obj_pcd,get_args,repeat_label,setup_seed


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
    mod = ppo_module.MLP
    cfg['environment']['get_pcd'] = False
    cfg['environment']['extra_dim'] = 1
elif cfg['module'] == 'mcg':
    mod = ppo_module.mcg_pcd
    cfg['environment']['get_pcd'] = True
    cfg['environment']['extra_dim'] = 301
cfg['seed']=args.seed

### get experiment parameters

pre_grasp_steps = cfg['environment']['pre_grasp_steps']
trail_steps = cfg['environment']['trail_steps']
test_inference = args.test
train_obj_id = args.obj_id
all_obj_train = True if args.all_objects else False
meta_info_dim = 4

### get network parameters
num_repeats= args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh
# if args.test:
#     dict_labels=joblib.load("raisimGymTorch/data/dexycb_test_labels.pkl")
# else:
#     dict_labels = joblib.load("raisimGymTorch/data/dexycb_grasptta_train.pkl")
#     # get the first row of array in dict_labels.
#     # The structure of dict_labels is {1:{'a']:array, 'b':array}, 2:{'a']:array, 'b':array}}
#     # here is the code
#     for key in dict_labels:
#         for key2 in dict_labels[key]:
#             dict_labels[key][key2] = dict_labels[key][key2][[0]]
#
#
dict_labels = joblib.load("raisimGymTorch/data/test.pkl")
for key in dict_labels:
    #if key!=1:continue
    for key2 in dict_labels[key]:
        dict_labels[key][key2] = dict_labels[key][key2][-10:]

#dict_labels = joblib.load("raisimGymTorch/data/test.pkl")
#dict_labels=joblib.load("raisimGymTorch/data/dexycb_test_labels.pkl")

if args.all_objects:
    dict_labels = concat_dict(dict_labels)
    repeated_label = repeat_label(dict_labels, 1)
else:
    repeated_label = repeat_label(dict_labels[args.obj_id], 1)

num_envs = repeated_label['final_qpos'].shape[0]
# mesh_path = "../rsc/meshes_simplified/008_pudding_box/mesh_aligned.obj"
# obj_pcd = get_obj_pcd(mesh_path)
obj_pcd = None

cfg['environment']['num_envs'] = 1 if args.vis_evaluate else num_envs
#obj_pcd = np.repeat(obj_pcd[np.newaxis, ...], cfg['environment']['num_envs'], 0)
cfg["testing"] = True if test_inference else False


env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'],label=repeated_label,obj_pcd=obj_pcd)
ob_dim = env.obsdim_for_agent
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps

avg_rewards = []

### Set up logging
log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + args.exp_name
saver = ConfigurationSaver(log_dir = log_dir,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=True)

ppo = get_ppo(mod)

### Loading a pretrained model
load_param(saver.data_dir.split('eval')[0]+args.weight, env, ppo.actor, ppo.critic, ppo.optimizer, saver.data_dir,args.cfg, store_again=False)


### Evaluate trained model visually (note always the first environment gets visualized)
if args.vis_evaluate:
    ### Start recording

    for i in range(num_envs):
        if args.store_video:
            env.turn_on_visualization()
            env.start_video_recording(
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + str(i) + '.mp4')
        ### Set labels and load objects for current label (only one visualization per rollout possible)

        set_guide=False
        env.move_to_first(i)

        next_obs,info = env.reset(add_noise=False)
        time.sleep(1)
        for step in range(n_steps):
            obs = next_obs
            ### Get action from policy
            action_pred = ppo.actor.architecture(torch.from_numpy(obs).to(device))
            frame_start = time.time()

            action_ll = action_pred.cpu().detach().numpy()
            ### After grasp is established remove surface and test stability
            if step>grasp_steps:
                if not set_guide:
                    env.set_root_control()
                    set_guide=True

            next_obs,reward, dones,info = env.step(action_ll)

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        ### Store recording
        if args.store_video:
            print('store video')
            env.stop_video_recording()
            env.turn_off_visualization()

### quantitative evaluation
else:
    disp_list, slipped_list, contact_ratio_list = [], [], []
    qpos_list, joint_pos_list, obj_pose_list = [], [], []

    set_guide=False

    next_obs, info = env.reset(add_noise=False)
    for step in range(n_steps):
        obs = next_obs
        action_ll = ppo.actor.architecture(torch.from_numpy(obs).to(device))
        frame_start = time.time()

        ### After grasp is established remove surface and test stability
        if step>grasp_steps and not set_guide:
            meta_info = info['meta_info']
            obj_pos_fixed = meta_info[:,-4:-1].copy()
            env.set_root_control()
            set_guide=True
        ### Record slipping and displacement
        if step>(grasp_steps+1):
            meta_info = info['meta_info']
            slipped_list.append(meta_info[:,-1].copy())
            obj_disp = np.linalg.norm(obj_pos_fixed-meta_info[:,-4:-1],axis=-1)
            disp_list.append(obj_disp)
            obj_pos_fixed = meta_info[:,-4:-1].copy()

        next_obs,reward, dones,info = env.step(action_ll.cpu().detach().numpy())

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)
    obj_idx_stacked = repeated_label['obj_idx_stacked']
    ### Log quantiative results
    for obj_id in np.unique(obj_idx_stacked):
        train_obj_id = obj_id + 1

        ### compute testing window
        sim_dt = cfg['environment']['simulation_dt']
        control_dt = cfg['environment']['control_dt']
        control_steps = int(control_dt / sim_dt)
        sim_to_real_steps = 1/(control_steps * sim_dt)
        window_5s = int(5*sim_to_real_steps)

        obj_idx_array = np.where(obj_idx_stacked == obj_id)[0]

        slipped_array = np.array(slipped_list)[:].transpose()[obj_idx_array]
        disp_array = np.array(disp_list)[:].transpose()[obj_idx_array]

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("object: ", obj_id+1))
        print('{:<40} {:>6}'.format("success: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')

        if not all_obj_train:
            np.save(log_dir+'/success_idxs',success_idx)


    ### Log average success rate over all objects
    if all_obj_train:
        slipped_array = np.array(slipped_list)[:].transpose()
        disp_array = np.array(disp_list)[:].transpose()

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        if len(success_idx) > 0:
            np.save(log_dir+'/success_idxs',success_idx)

        print('----------------------------------------------------')
        print('{:<40}'.format("all objects"))
        print('{:<40} {:>6}'.format("total success rate: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')



