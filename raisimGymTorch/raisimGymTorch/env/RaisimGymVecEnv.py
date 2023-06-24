# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import numpy as np
import platform
import os
import copy
from scipy.spatial.transform import Rotation as R
from raisimGymTorch.helper.utils import dgrasp_to_mano,show_pointcloud_objhand,IDX_TO_OBJ
import torch
from manopth.manolayer import ManoLayer
from raisimGymTorch.helper.utils import IDX_TO_OBJ, get_obj_pcd
from raisimGymTorch.algo.ppo.module import PointNetAutoEncoder
import trimesh

class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=False,
                 seed=0, normalize_rew=True, clip_obs=10., label=None,
                 obj_pcd=None):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        model = PointNetAutoEncoder.load_from_checkpoint(
            'raisimGymTorch/data_all/pointnet_ae.ckpt')

        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.obsdim_for_agent = self.wrapper.getObDim()+cfg['extra_dim']-cfg['metainfo_dim']
        self.meta_dim = cfg['metainfo_dim']
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.wrapper.getObDim()], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.obsdim_for_agent])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.label = label
        self.time_step = 0
        self.n_steps =  cfg['pre_grasp_steps']+ cfg['trail_steps']
        # if obj_pcd:

        label['final_contact_pos'] = np.zeros_like(label['final_contacts'])
        self.load_object(label['obj_name'], label['obj_w_stacked'], label['obj_dim_stacked'],
                         label['obj_type_stacked'])
        self.set_goals(label['final_obj_pos'], label['final_ee'], label['final_pose'], label['final_contact_pos'],
                       label['final_contacts'])
        self.get_pcd = cfg['get_pcd']
        self.mano_layer = ManoLayer(mano_root='raisimGymTorch/data', flat_hand_mean=True, ncomps=45, use_pca=False).to('cuda')
        mano_layer = ManoLayer(mano_root='raisimGymTorch/data', flat_hand_mean=False, ncomps=45, use_pca=True)
        mean_axisang = mano_layer.th_hands_mean
        self.th_comps = mano_layer.th_comps.numpy()

        mean_pca = torch.einsum('bi,ij->bj', [mean_axisang, torch.inverse(mano_layer.th_comps)])

        self.mean_pca = mean_pca.numpy()

        if self.get_pcd:
            self.obj_pcd = np.zeros([self.num_envs,3000,3])
            self.obj_normal = np.zeros([self.num_envs,3000,3])
            self.obj_mesh = {}
            for obj_name in np.unique(label['obj_name']):
                p = f'../rsc/meshes_simplified/{obj_name}/textured_simple.obj'
                #obj_pcd = get_obj_pcd(p,num_p=3000)
                obj_mesh = trimesh.load_mesh(p)
                sampled_points, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
                sampled_normals = obj_mesh.face_normals[face_id]
                obj_num = np.sum(label['obj_name']==obj_name)
                obj_pcd = np.repeat(sampled_points[np.newaxis, ...], obj_num, 0)
                idx = np.where(label['obj_name']==obj_name)[0]
                self.obj_pcd[idx] = obj_pcd
                self.obj_normal[idx] = sampled_normals
                for i in idx:
                    self.obj_mesh[i] = obj_mesh

            inp = torch.tensor(self.obj_pcd).float().permute(0, 2, 1)
            self.obj_embed = model.encoder(inp).cpu().detach().numpy()
            self.obj_pcd = torch.tensor(self.obj_pcd,device='cuda').float()

    def move_to_first(self,i):

        for k,v in self.label.items():
            label_to_move = v[i].copy()
            self.label[k][0] = label_to_move
        label = self.label
        self.load_object(label['obj_name'], label['obj_w_stacked'], label['obj_dim_stacked'],
                         label['obj_type_stacked'])
        self.set_goals(label['final_obj_pos'], label['final_ee'], label['final_pose'], label['final_contact_pos'],
                       label['final_contacts'])

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.time_step+=1
        self.wrapper.step(action, self._reward, self._done)

        obs, info = self.observe()

        reward, reward_info = self.get_reward_info(obs)

        info['reward_info'] = reward_info

        return obs, reward, self._done.copy(), info

    def get_reward_info(self,obs):
        pca_reward = self.get_pca_rewards(obs)
        reward_info = self.wrapper.getRewardInfo()
        for i in range(len(reward_info)):
            reward_info[i]['pca'] = pca_reward[i]
            reward_info[i]['reward_sum'] += reward_info[i]['pca']
        reward = np.array([r['reward_sum'] for r in reward_info])
        return reward, reward_info


    def get_pca_rewards(self, obs):
        bs = obs.shape[0]
        mano_param = dgrasp_to_mano(obs[:, :51])

        joint_pca = np.matmul(mano_param[:,3:48], np.linalg.inv(self.th_comps))
        joint_pca_norm = joint_pca / np.linalg.norm(joint_pca, axis=-1, keepdims=True)

        pca_target = self.mean_pca.repeat(bs, 0)
        pca_target_norm = pca_target / np.linalg.norm(pca_target, axis=-1, keepdims=True)

        pca_dist = joint_pca_norm * pca_target_norm
        cos_sim = pca_dist.sum(-1)-1
        cos_sim[cos_sim>-0.4] *= 0.1
        return cos_sim

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        #ret_obs = {}
        self.wrapper.observe(self._observation)
        ob_dim = self._observation.shape[-1]
        meta_info = self._observation[:,ob_dim-self.meta_dim:].copy()
        obs = self._observation[:,:ob_dim-self.meta_dim].copy()

        # ret_obs['hand_obs'] = obs[:,:121]
        # ret_obs['label_obs'] = obs[:,121:264]
        # ret_obs['obj_info'] = obs[:,264:]

        step_obs = np.zeros([self._observation.shape[0],1]).astype('float32')
        step_obs[:] = self.time_step/self.n_steps
        obs = np.concatenate([obs,step_obs],axis=1)


        if self.get_pcd:
            # obj_pcd = self.obj_pcd
            # env_num, pcd_num, dim = obj_pcd.shape
            #
            obj_pos = copy.copy(obs[:, 264:267])
            obj_euler = copy.copy(obs[:, 267:270])
            hand_pos = copy.copy(obs[:, :3])
            hand_euler = copy.copy(obs[:, 3:6])
            #
            # r_obj = obj_euler[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            # obj_pos = obj_pos[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            # r_obj = R.from_euler('XYZ', r_obj, degrees=False)
            #
            # obj_pcd = r_obj.apply(obj_pcd.reshape(-1, dim)) - obj_pos
            # #obj_pcd = obj_pcd.reshape(env_num, -1).astype('float32')
            #
            # r_hand = hand_rot[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            # hand_pos = hand_pos[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            # r_hand = R.from_euler('XYZ', r_hand, degrees=False)
            #
            # obj_pcd = r_hand.apply(obj_pcd.reshape(-1, dim)) + hand_pos
            # obj_pcd = obj_pcd.reshape(env_num, -1).astype('float32')

            # Convert the Euler angles to rotation matrices
            hand_rot_mat = R.from_euler('XYZ', hand_euler, degrees=False)

            # Transform obj_pos from the hand frame to the world frame
            obj_pos_world = hand_rot_mat.apply(obj_pos) + hand_pos

            # Calculate the position of the hand in the object's frame
            hand_pos_in_obj_frame = hand_pos - obj_pos_world

            # Transform obj_euler from the hand frame to the world frame
            obj_rot_mat_hand = R.from_euler('XYZ', obj_euler, degrees=False)
            obj_rot_mat_world = hand_rot_mat * obj_rot_mat_hand

            # Calculate the rotation matrix for the hand in the object's frame
            hand_rot_in_obj_frame = obj_rot_mat_world.inv() * hand_rot_mat

            # Convert the rotation matrix back to Euler angles
            hand_euler_in_obj_frame = hand_rot_in_obj_frame.as_euler('XYZ', degrees=False)

            hand_pos_in_obj_frame = hand_rot_in_obj_frame.apply(hand_pos_in_obj_frame)

            gc = copy.copy(obs[:, :51])
            gc[:, :3] = hand_pos_in_obj_frame
            gc[:, 3:6] = hand_euler_in_obj_frame
            mano_param = dgrasp_to_mano(gc)
            mano_param = torch.from_numpy(mano_param).float().to('cuda')

            verts, joints = self.mano_layer(th_pose_coeffs=mano_param[:, :48], th_trans=mano_param[:, -3:])
            #hand_face = self.mano_layer.th_faces
            verts /= 1000
            joints /= 1000
            # get nearest point between joints and obj_pcd
            obj_pcd = self.obj_pcd

            dists = torch.cdist(joints, obj_pcd)

            # Get the minimum distance and index
            C, D = torch.min(dists, dim=2)
            points = torch.gather(obj_pcd, 1, D.unsqueeze(2).expand(-1, -1, 3))
            #normals = torch.gather(self.obj_normal, 1, D.unsqueeze(2).expand(-1, -1, 3))

            points_and_dist = torch.cat((points, C.unsqueeze(2)), dim=2).reshape(points.shape[0], -1).cpu().detach().numpy()
            joints = joints.reshape(points.shape[0], -1).cpu().detach().numpy()
            add_obs = np.concatenate((joints, points_and_dist,self.obj_embed), axis=1)
            # if self.time_step%50==0:
            #     show_pointcloud_objhand(verts[0], self.obj_pcd[0].reshape(-1, 3))
            #obs = np.concatenate([obs, self.obj_embed], axis=-1)
            obs = np.concatenate([obs, add_obs], axis=-1)

        info = {}
        info['meta_info'] = meta_info
        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation)
        else:
            return obs, info

    def set_root_control(self):
        self.wrapper.set_root_control()

    def reset(self, add_noise=True):
        self.time_step = 0
        qpos_reset = self.label['qpos_reset'].copy()
        obj_pose_reset = self.label['obj_pose_reset'].copy()
        num_envs = qpos_reset.shape[0]
        if add_noise:
            random_noise_pos = np.random.uniform([-0.02, -0.02, 0.01], [0.02, 0.02, 0.01], (num_envs, 3)).copy()
            random_noise_qpos = np.random.uniform(-0.05, 0.05, (num_envs, 48)).copy()
            qpos_noisy_reset = qpos_reset
            qpos_noisy_reset[:, :3] += random_noise_pos[:, :3]
            qpos_noisy_reset[:, 3:] += random_noise_qpos[:, :]
            ### Run episode rollouts
            self.reset_state(qpos_noisy_reset, np.zeros((num_envs, 51), 'float32'), obj_pose_reset)
        else:
            self.reset_state(qpos_reset, np.zeros((num_envs, 51), 'float32'), obj_pose_reset)

        obs, info = self.observe()

        reward, reward_info = self.get_reward_info(obs)
        info['reward_info'] = reward_info
        return obs, info

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
        #obj_idx = [IDX_TO_OBJ[obj_id+1][0] for obj_id in obj_idx]
        obj_idx = [str(obj_id) for obj_id in obj_idx]
        obj_type = obj_type.astype('int32')
        self.wrapper.load_object(obj_idx, obj_weight, obj_dim, obj_type)

    def reset_state(self, init_state, init_vel, obj_pose):

        self.wrapper.reset_state(init_state, init_vel, obj_pose)

    def set_goals(self, obj_pos, ee_pos, pose, contact_pos, normals):
        self.wrapper.set_goals(obj_pos, ee_pos, pose, contact_pos, normals)

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
