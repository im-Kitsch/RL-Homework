import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import quanser_robots

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartpoleStabShort-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        self.affine1 = nn.Linear(5, 128)
        self.affine2 = nn.Linear(128, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.stddev = torch.tensor([10.0])

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)


class ValueNN(nn.Module):
    def __init__(self):
        super(ValueNN, self).__init__()
        self.lin1 = nn.Linear(5, 128)
        self.head = nn.Linear(128, 1)

        self.states = []

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.head(x)


policy_nn = PolicyNN()
value_nn = ValueNN()
policy_optimizer = optim.Adam(policy_nn.parameters(), lr=1e-900)
value_optimizer = optim.Adam(value_nn.parameters(), lr=1e-600)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    mean = policy_nn(torch.Tensor([state]))
    stddev = policy_nn.stddev
    m = Normal(mean, stddev)
    action = m.sample()
    policy_nn.saved_log_probs.append(m.log_prob(action))
    return np.array([action.item()])


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    advantage_values = []

    for r in policy_nn.rewards[::-1]:
        R = r + args.gamma * R
        R = torch.Tensor(R)
        rewards.insert(0, R)

    for tot_disc_rwd, s in zip(rewards, value_nn.states[:]):
        value = policy_nn(torch.Tensor(s))
        advantage_values += [tot_disc_rwd - value]


    for log_prob, advantage in zip(policy_nn.saved_log_probs, advantage_values):
        policy_loss.append(-log_prob * advantage)

    value_loss = torch.tensor(advantage_values, requires_grad=True)

    value_optimizer.zero_grad()
    value_loss = value_loss.sum()
    value_loss.backward()
    value_optimizer.step()

    policy_loss = torch.stack(policy_loss)
    policy_loss = policy_loss.sum()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    del policy_nn.rewards[:]
    del policy_nn.saved_log_probs[:]
    del value_nn.states[:]


def main():
    reward_threshold = 190
    pretty_good = False
    running_reward = 10
    for i_episode in count(1):
        episode_rwd = 0
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            value_nn.states.append(torch.Tensor(state))
            state, reward, done, _ = env.step(action)
            episode_rwd += reward
            if pretty_good == True or i_episode>1000:
                env.render()
            policy_nn.rewards.append(torch.Tensor(np.array([reward])))
            if done:
                break
        # print(episode_rwd)

        # print(policy_nn.stddev)
        if policy_nn.stddev > 0.5:
            policy_nn.stddev /= 1.01
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\t\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > reward_threshold:
            print('pretty good')
            pretty_good = True



if __name__ == '__main__':
    main()
