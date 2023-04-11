from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import gym
import numpy as np
from stable_baselines3 import DQN

class SegmentWrapper(gym.ObservationWrapper):
    def __init__(self, env, nb_max_mask=15):
        super().__init__(env)
        sam = sam_model_registry["vit_b"](checkpoint="models/superlight_model.pth") #76 sec per inf
        # sam = sam_model_registry["default"](checkpoint="full_model.pth") #100 sec per inf
        # sam = sam_model_registry["vit_l"](checkpoint="light_model.pth") #90 sec per inf # BBAAAAAAAAD
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.nb_max_mask = nb_max_mask
        self.observation_space = gym.spaces.Box(shape=(4 * self.nb_max_mask, 4), low=0, high=84)

    def observation(self, obs):
        mask_obs = np.zeros((4 * self.nb_max_mask, 4))
        for i, im in enumerate(obs):
            im_gray = im.T[0]
            im_bgr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2BGR)
            masks = self.mask_generator.generate(im_bgr)
            for j in range(min(len(masks),self.nb_max_mask)):
                mask_obs[j*4:(j+1)*4,i] = masks[j]["bbox"]
        return mask_obs


def seed_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = AtariWrapper(env)
        env = SegmentWrapper(env)
        return env
    set_random_seed(seed)
    return _init


num_cpu = 4
# There already exists an environment generator that will make and wrap atari environments correctly.
env = DummyVecEnv([seed_env('PongNoFrameskip-v4', i) for i in range(num_cpu)])
model = DQN("MlpPolicy", env, verbose=1, learning_starts = 0)
model.learn(total_timesteps=100000, log_interval=4)
model.save("segmented_pong_dqn")
# s = env.reset()
# done = False
# while not done:
#     a = env.action_space.sample()
#     s,r,done, info = env.step([a])
#     print(s, r)
