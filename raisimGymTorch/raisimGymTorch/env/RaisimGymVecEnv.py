# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import numpy as np
import platform
import os
import copy
from scipy.spatial.transform import Rotation as R

class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=False, seed=0, normalize_rew=True, clip_obs=10., label=None,
                 obj_pcd=None):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        self.obj_pcd = obj_pcd
        self.load_object(label['obj_idx_stacked'], label['obj_w_stacked'], label['obj_dim_stacked'],
                         label['obj_type_stacked'])
        self.set_goals(label['final_obj_pos'], label['final_ee'], label['final_pose'], label['final_contact_pos'],
                       label['final_contacts'])
        self.get_pcd = cfg['get_pcd']

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
        self.wrapper.load_object(obj_idx, obj_weight, obj_dim, obj_type)

    def move_to_first(self,i):

        for k,v in self.label.items():
            label_to_move = v[i].copy()
            self.label[k][0] = label_to_move
        label = self.label

        self.wrapper.set_goals(label['final_obj_pos'], label['final_ee'], label['final_pose'], label['final_contact_pos'],
                       label['final_contacts'])

    def set_goals(self, obj_pos, ee_pos, pose, contact_pos, normals):
        self.wrapper.set_goals(obj_pos, ee_pos, pose, contact_pos, normals)

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
        return obs, self._reward.copy(), self._done.copy(), info

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
            obj_pcd = self.obj_pcd
            env_num, pcd_num, dim = obj_pcd.shape

            obj_pos = copy.copy(self._observation[:, -15:-12])
            obj_euler = copy.copy(self._observation[:, -12:-9])


            r_obj = obj_euler[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            obj_pos = obj_pos[:, np.newaxis].repeat(pcd_num, 1).reshape(-1, dim)
            r_obj = R.from_euler('XYZ', r_obj, degrees=False)

            obj_pcd = r_obj.apply(obj_pcd.reshape(-1, dim)) - obj_pos
            obj_pcd = obj_pcd.reshape(env_num, -1).astype('float32')
            obs = np.concatenate([obs, obj_pcd], axis=-1)
            #ret_obs['obj_pcd'] = obj_pcd.reshape(env_num,-1,3).astype('float32')
            # gc = copy.copy(self._observation[:,:51])
            # idx = 11
            # verts,joints = dgrasp_to_mano(self._observation[idx,:51])
            # show_pointcloud_objhand(verts,obj_pcd[idx].reshape(-1,3))


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
        return obs, info

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
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
