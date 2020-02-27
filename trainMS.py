import gym
import numpy as np
import torch
import os
from itertools import count

from baselines.common.vec_env import SubprocVecEnv
from baselines.common import set_global_seeds
from Model import MultiSensorSimple
from ppo import PPO



modelpath = './ppoCNN_MS_model.pth'
play_mode = False
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
num_envs = 2
env_name = "gym_chrono.envs:camera_obstacle_avoidance-v1"
disp_plot = False
hidden_size      = 256
lr               = 3e-4
# Steps per env
num_steps        = 100
mini_batch_size  = num_steps/20
ppo_epochs       = 8
threshold_reward = 5000
save_interval = 10
max_frames = np.inf
max_pol_updates = 200
test_interval = 0.5
test_interval = 1e12
do_test = False
increasing_length = 0

def make_env(env_id, rank, seed=0):
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
        return env
    set_global_seeds(seed)
    return _init





if __name__ == "__main__":
    envs = SubprocVecEnv([make_env(env_name, i) for i in range(num_envs)])
    env = gym.make(env_name)
    tuple_ob = isinstance(env.observation_space , gym.spaces.Tuple)

    #num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    model = MultiSensorSimple([45,80], 2, num_outputs).to(device)
    if os.path.isfile(modelpath):
        model.load_state_dict(torch.load(modelpath))

    ppo = PPO(model=model, envs = envs, device = device,  lr = lr, modelpath = modelpath, tuple_ob = tuple_ob)
    if not play_mode:
        ppo.ppo_train(num_steps, mini_batch_size, ppo_epochs,
                  max_frames, max_pol_updates,save_interval, increasing_length,
                  test_interval,do_test, threshold_reward, disp_plot, env)


    # <h1>Saving trajectories for GAIL</h1>
    max_expert_num = 50000
    num_steps = 0
    #expert_traj = []

    for i_episode in count():
        if not play_mode:
            break
        state = env.reset()
        env.play_mode = True
        done = False
        total_reward = 0

        while not done:
            if tuple_ob:
                state = [torch.FloatTensor(s).unsqueeze(0).to(device) for s in state]
            else:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            #expert_traj.append(np.hstack([state, action]))
            num_steps += 1
            env.render()

        print("episode:", i_episode, "reward:", total_reward)

        if num_steps >= max_expert_num:
            break
"""
    if play_mode:

        expert_traj = np.stack(expert_traj)
        print()
        print(expert_traj.shape)
        print()
        np.save("expert_traj.npy", expert_traj)
"""