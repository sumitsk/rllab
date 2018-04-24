from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        self.velocity_dir = 'posx'
        self.penalty = 1.0                
        self.use_gym_obs = False
        self.use_gym_reward = False

        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_forward_reward_rllab(self):
        #print(self.velocity_dir)
        
        comvel = self.get_body_comvel("torso")
        if self.velocity_dir == 'posx':
            forward_reward = comvel[0] -self.penalty * abs(comvel[1])
        elif self.velocity_dir == 'posy':
            forward_reward = comvel[1] -self.penalty * abs(comvel[0])
        elif self.velocity_dir == 'negx':
            forward_reward = -comvel[0] - self.penalty * abs(comvel[1])
        elif self.velocity_dir == 'negy':
            forward_reward = -comvel[1] - self.penalty * abs(comvel[0])
        else:
            raise NotImplementedError
        return forward_reward

    def get_forward_reward_gym(self, posbefore, posafter):
        vel = (posafter - posbefore) / 0.05

        if self.velocity_dir == 'posx':
            forward_reward = vel[0]
        elif self.velocity_dir == 'posy':
            forward_reward = vel[1] 
        elif self.velocity_dir == 'negx':
            forward_reward = -vel[0] 
        elif self.velocity_dir == 'negy':
            forward_reward = -vel[1] 
        else:
            raise NotImplementedError
        return forward_reward
     

    def get_current_obs(self):
        if self.use_gym_obs:
            return np.concatenate([
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ])
            
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)
        
    def step(self, action):
        posbefore = self.get_body_com("torso")
        self.forward_dynamics(action)
        posafter = self.get_body_com("torso")

        '''
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        '''
        if self.use_gym_reward:
            forward_reward = self.get_forward_reward_gym(posbefore, posafter)
            ctrl_cost = .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-3 * np.sum(
                np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0

        else:    
            forward_reward = self.get_forward_reward_rllab()
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
            contact_cost = 0.5 * 1e-3 * np.sum(
                np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
            survive_reward = 0.05
        
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

