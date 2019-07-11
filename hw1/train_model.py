import os
import pickle
import numpy as np

import argparse
import models
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def save_checkpoint(state, filename):
    torch.save(state, filename)

def train_validate(model, dataloader, optim, loss_fn, epoch,train):
    loss_all = 0
    model.train() if train else model.eval()
    for batch_idx, (x, y) in enumerate(dataloader):

        pred_act = model(x)
        loss = loss_fn(pred_act, y)

        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item() / len(x)))



        loss_all += loss.item()
    return loss_all / len(dataloader.dataset)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=100)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()



    #
    expert_data = pickle.load(open(os.path.join('expert_data/' + args.envname + '.pkl'), 'rb'))
    expert_obs = expert_data['observations']
    expert_act = expert_data['actions']

    print(expert_obs.shape)
    print(expert_act.shape)

    expert_obs_train, expert_obs_test, expert_act_train, expert_act_test = train_test_split(expert_obs, expert_act, test_size = 0.2)

    print(expert_obs_train.shape)
    print(expert_act_train.shape)
    print(expert_obs_test.shape)
    print(expert_act_test.shape)

    expert_obs_train = torch.from_numpy(expert_obs_train).type(torch.FloatTensor)
    expert_act_train = torch.from_numpy(expert_act_train).squeeze(1).type(torch.FloatTensor)
    expert_obs_test = torch.from_numpy(expert_obs_test).type(torch.FloatTensor)
    expert_act_test = torch.from_numpy(expert_act_test).squeeze(1).type(torch.FloatTensor)
    dataset_train = TensorDataset(expert_obs_train, expert_act_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = TensorDataset(expert_obs_test, expert_act_test)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)





    bc_model = models.bc(expert_obs.shape[1], expert_act.shape[1])

    optim = Adam(bc_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    n_epochs = 50


    train_loss = []
    val_loss = []
    best_loss = np.inf

    for epoch in range(0, n_epochs):
        t_loss = train_validate(bc_model, dataloader_train, optim, loss_fn, epoch, train=True)
        train_loss.append(t_loss)
        v_loss = train_validate(bc_model, dataloader_val, optim, loss_fn, epoch, train=False)
        val_loss.append(v_loss)
        print('Train Epoch: {} \t train loss: {:.6f} \t val loss: {:.6f} '.format(
            epoch, t_loss, v_loss))
        if v_loss < best_loss:
            best_loss = v_loss
            print('Writing model checkpoint')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': bc_model.state_dict(),
                'val_loss': v_loss
            },
                'models/bc_{}.pt'.format(args.envname))

    plt.plot(np.arange(len(train_loss)), train_loss, label='train')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val')
    plt.title('Training')
    plt.xlabel('Step')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/loss_'+ args.envname + '.png')
    plt.show()







if __name__ == '__main__':
    main()
