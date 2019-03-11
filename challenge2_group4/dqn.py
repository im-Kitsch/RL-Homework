import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from supplementary import DQNMemory
from supplementary import DQN


class DQNAlgorithmTraining:

    def __init__(self, env, steps_per_episode=10000):
        # I put this really high because i had the problem, that it always 'forgot', what it learned in the beginning,
        # causing it to get a really bad performance after a peak 100 episodes (1 million timesteps earlier)
        self.capacity = 5000000  # capacity of memory
        self.batch_size = 500  # Size of samples for learning
        self.memory = DQNMemory(self.capacity)  # Initialize Memory
        self.env = env  # Environment
        self.epsilon = 0.9  # Start random factor
        self.gamma = 0.98  # Gamma for optimization
        self.episodes = 1000  # Episodes for learning
        self.steps_per_episode = steps_per_episode  # Time each episode

        self.rwd_overall = 0
        self.rwd_episodes_len = 20
        self.rwd_episodes = np.zeros(self.rwd_episodes_len)
        self.rwd_episodes_array = np.empty(0)
        self.rwd_overall_array = np.empty(0)

        self.min_vol = max(env.action_space.low[0], -20)  # Minimal voltage to apply
        self.max_vol = min(env.action_space.high[0], 20)  # Maximal voltage to apply
        self.observation_space = env.observation_space.shape[0]  # Number of observations per observation
        self.nn_outputs = 20  # Number of NeuralNetwork outputs, has to be > 1
        self.action_space = np.arange(
            self.min_vol, self.max_vol + 1,
                          (self.max_vol - self.min_vol) / (self.nn_outputs - 1),
            dtype=float
        )  # Discrete ActionSpace

        self.policy_nn = DQN(self.observation_space, self.nn_outputs)  # Initialise Policy_Network

        self.target_nn = DQN(self.observation_space, self.nn_outputs)

        self.target_nn.load_state_dict(self.policy_nn.state_dict())  # Target_Network = Policy_Network

        self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')  # Define Loss-Function
        self.learning_rate = 0.0001  # Set learning rate

        self.update_target = 200  # Update target_network after n iterations
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=self.learning_rate)  # Define optimizer

    def main(self):

        plt.ion()

        for episode in range(1, self.episodes):
            print('episode: ' + str(episode))  # Print Episode
            # for param in self.policy_nn.parameters():
            #    print(param)
            self.obs = self.env.reset()  # Reset environment
            rwd_sum = 0  # accumulated reward
            for t in range(self.steps_per_episode):
                act = self.get_action()  # Select action by random or policy

                # Higher Randomness in early epochs, lower randomness, when nn is trained

                # Perform step on environment
                obs, rwd, done, _ = self.env.step(np.array([self.action_space[act]]))  # perform action on environment

                self.memory.push(self.obs, act, rwd, obs, done)  # Adds to memory
                self.obs = obs  # new_obs = old_obs

                rwd_sum += rwd  # Update accumulated reward

                self.optimize_model()  # Perform optimization step

                if done:  # If terminated
                    print('Done')
                    break  # stop episode

                if t % self.update_target == 0:
                    self.target_nn.load_state_dict(self.policy_nn.state_dict())

                if episode % 20 == 0:
                    if t % 10 == 0:
                        self.env.render()

            if self.epsilon >= 0.01:
                self.epsilon /= 1.01
            print(self.epsilon)
            self.rwd_overall += rwd_sum
            self.rwd_episodes[episode % self.rwd_episodes_len] = rwd_sum

            if episode % self.rwd_episodes_len + 1 == self.rwd_episodes_len:
                print('We are at step: ' + str(episode))
                self.plot(episode)

            self.optimize_model()

            print(episode)
            model_states = {
                'episode': episode,
                'model_state_dict': self.policy_nn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(model_states,
                       f"supplementary/challenge2_training_results.pt")
        print('Complete')
        self.env.render()
        self.env.close()

        return self.get_policy_action

    def get_action(self):
        if random.random() < self.epsilon:
            return random.randrange(0, self.nn_outputs)  # Get random action
        else:
            _, pos = torch.max(self.policy_nn(torch.Tensor(self.obs)), 0)  # Get action by policy
            return pos.item()

    def get_policy_action(self, obs):
        _, pos = torch.max(self.policy_nn(torch.Tensor(obs)), 0)  # Get action by policy
        act = pos.item()
        return np.array([self.action_space[act]])

    def optimize_model(self):
        # sample:
        # [0] = observation
        # [1] = action
        # [2] = reward
        # [3] = new_observation
        # [4] = done

        samples = self.memory.sample(self.batch_size)
        tensor_samples = torch.stack(samples)

        obs = tensor_samples[:, 0:self.observation_space]
        act = tensor_samples[:, self.observation_space:self.observation_space + 1]
        rwd = tensor_samples[:, self.observation_space + 1: self.observation_space + 2]
        new_obs = tensor_samples[:, self.observation_space + 2: 2 * self.observation_space + 2]
        done = tensor_samples[:, 2 * self.observation_space + 2: 2 * self.observation_space + 3]

        act_q, pos_q = torch.max(self.target_nn(new_obs), 1)
        act_q = act_q.view(-1, 1)
        y_j = (rwd + self.gamma * act_q)
        # if done == True set y_j to rwd
        # looks so complicated cause of vector form
        y_j = y_j * (torch.ones(len(y_j)).reshape(-1, 1) - done) + rwd * done

        policy_action_value = self.policy_nn(obs)
        indexing = (act).int().numpy()
        # This gets the n. action for the each actionsspace depending on observation in our policy
        # we need arange so
        q_now = policy_action_value[np.arange(len(policy_action_value)).reshape(-1, 1), indexing]

        loss = self.loss_fn(q_now, y_j)

        self.optimizer.zero_grad()  # set grad to zero

        loss.backward()  # Perform backward loss

        self.optimizer.step()  # Perform optimizer step

    def plot(self, episode):
        self.rwd_episodes_array = np.append(self.rwd_episodes_array, np.mean(self.rwd_episodes))
        self.rwd_overall_array = np.append(self.rwd_overall_array, self.rwd_overall / episode)
        print('Mean reward after ' + str(
            self.rwd_episodes.__len__()) + ' Episodes: ' +
              str(self.rwd_episodes_array[-1])
              )
        print('Reward overall: ' + str(self.rwd_overall_array[-1]))
        t = np.arange(0, len(self.rwd_episodes_array), 1)
        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(t * self.rwd_episodes_len + self.rwd_episodes_len, self.rwd_episodes_array, 'r--',
                 label='Last 20 Episodes')
        plt.plot(t * self.rwd_episodes_len + self.rwd_episodes_len, self.rwd_overall_array, 'k', label='Overall')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig(
            'supplementary/challenge2_training_results.png')
        self.rwd_episodes = np.zeros(self.rwd_episodes_len)
