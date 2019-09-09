import gym
from gym import wrappers
import os.path as osp
import random

import torch
import torch.nn as nn

import dqn_pt
from dqn_utils_pt import *
from atari_wrappers import *


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2DRelu(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2DRelu, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super(Conv2DRelu, self).forward(x)
        return self.relu(x)


class AtariModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AtariModel, self).__init__()
        self.c1 = Conv2DRelu(input_shape, 32, 8, 4, 1)
        self.c2 = Conv2DRelu(32, 64, 4, 2, 1)
        self.c3 = Conv2DRelu(64, 64, 3, 1, 1)
        self.bf = BatchFlatten()
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.bf(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def atari_learn(env, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    optimizer = dqn_pt.OptimizerSpec(
        constructor=torch.optim.Adam,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )


    dqn_pt.QLearner.learn(
        env=env,
        q_func=AtariModel,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=True
    )

    env.close()


def set_global_seeds(i):
    torch.manual_seed(i)


def get_env(task, seed):
    env = gym.make('PongNoFrameskip-v4')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env


def main():
    # Get Atari games.
    task = gym.make('PongNoFrameskip-v4')

    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)

    env = get_env(task, seed)
    atari_learn(env, num_timesteps=2e8)


if __name__ == "__main__":
    main()
