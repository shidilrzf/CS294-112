import os
import pickle
import numpy as np
import gym
import load_policy_pytorch
import argparse
import models
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--expert_policy_file', type=str)
parser.add_argument('--envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int, default=100)
parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
parser.add_argument('--batch_size', type=int, default=20)

args = parser.parse_args()

expert_data = pickle.load(open(os.path.join('expert_data/' + args.envname + '.pkl'), 'rb'))
expert_obs = expert_data['observations']
expert_act = expert_data['actions']

model = models.bc(expert_obs.shape[1], expert_act.shape[1])
path_model = 'models/bc_{}.pt'.format(args.envname)
checkpoint=torch.load(path_model, map_location='cpu' )
model.load_state_dict(checkpoint['state_dict'])
print('loading and building expert policy')
policy_fn = load_policy_pytorch.load_policy(args.expert_policy_file)
print('loaded and built')

env = gym.make(args.envname)
max_steps = args.max_timesteps or env.spec.timestep_limit


returns = []
observations = []
actions = []

for i in range(args.num_rollouts):
        #print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = policy_fn(obs)
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if args.render:
            env.render()
            #if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if steps >= max_steps:
            break
    returns.append(totalr)

print('returns', returns)
print('mean return', np.mean(returns))
print('std of return', np.std(returns))

returns_bc = []
observations = []
actions = []

for i in range(args.num_rollouts):
        #print('iter', i)
    obs = env.reset()
    done = False
    totalr_bc = 0.
    steps = 0
    while not done:
        obs_tensor = torch.from_numpy(obs).type(torch.FloatTensor)
        action = model(obs_tensor)
        action = action.detach().numpy()
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr_bc += r
        steps += 1
        if args.render:
            env.render()
            #if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if steps >= max_steps:
            break
    returns_bc.append(totalr_bc)

print('returns', returns_bc)
print('mean return', np.mean(returns_bc))
print('std of return', np.std(returns_bc))


