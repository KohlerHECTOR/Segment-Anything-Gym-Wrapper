from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import gym
import numpy as np
from stable_baselines3 import DQN

class SegmentWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask_generator, nb_max_mask=15):
        super().__init__(env)
        self.mask_generator = mask_generator
        self.nb_max_mask = nb_max_mask
        self.observation_space = gym.spaces.Box(shape=(4 * self.nb_max_mask, 1), low=0, high=84)

    def observation(self, obs):
        mask_obs = np.zeros((4 * self.nb_max_mask, 1))
        for i, im in enumerate(obs.T):
            im_gray = im.T[0]
            im_bgr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2BGR)
            masks = self.mask_generator.generate(im_bgr)
            for j in range(min(len(masks),self.nb_max_mask)):
                mask_obs[j*4:(j+1)*4,i] = masks[j]["bbox"]
        return mask_obs

class ArrayAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, act):
        return [act]

sam = sam_model_registry["vit_b"](checkpoint="models/superlight_model.pth") #76 sec per inf
# sam = sam_model_registry["default"](checkpoint="full_model.pth") #100 sec per inf
# sam = sam_model_registry["vit_l"](checkpoint="light_model.pth") #90 sec per inf # BBAAAAAAAAD

mask_generator = SamAutomaticMaskGenerator(sam)
# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env = SegmentWrapper(env, mask_generator)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = DQN("MlpPolicy", env, verbose=1, learning_starts = 0)
model.learn(total_timesteps=10000, log_interval=4)
model.save("segmented_pong_dqn")
# s = env.reset()
# done = False
# while not done:
#     a = env.action_space.sample()
#     s,r,done, info = env.step([a])
#     print(s, r)
