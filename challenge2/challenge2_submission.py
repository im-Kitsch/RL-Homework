"""
Submission template for Programming Challenge 2: Approximate Value Based Methods.


Fill in submission info and implement 4 functions:

- load_dqn_policy
- train_dqn_policy
- load_lspi_policy
- train_lspi_policy

Keep Monitor files generated by Gym while learning within your submission.
Example project structure:

challenge2_submission/
  - challenge2_submission.py
  - dqn.py
  - lspi.py
  - dqn_eval/
  - dqn_train/
  - lspi_eval/
  - lspi_train/
  - supplementary/

Directories `dqn_eval/`, `dqn_train/`, etc. are autogenerated by Gym (see below).
Put all additional results into the `supplementary` directory.

Performance of the policies returned by `load_xxx_policy` functions
will be evaluated and used to determine the winner of the challenge.
Learning progress and learning algorithms will be checked to confirm
correctness and fairness of implementation. Supplementary material
will be manually analyzed to identify outstanding submissions.
"""

import numpy as np

info = dict(
    group_number=4,  # change if you are an existing seminar/project group
    authors="Christopher Sura; Alexander Gübler; Zhiyuan Hu",
    description="For the swing up in the lab we usually set our boundaries to 5 Volt, which worked out alright "
                "and always converged to about 8000 reward, which is suboptimal for winning the challenge. "
                "When having 5 Volt, it only swings up when the exploration parameter gets really low "
                "so we decided to rather choose 20 Volt, where you can even swing up while having a high randomness "
                "with this when can focus on whats the best policy balancing the stick rather than just swinging up "
                "since this gives the most point, when having a long policy. "
                "When you have high Voltages we also need a lot of possible actions, since applying "
                "high Voltages decrease the reward. "
                "So in order to make a high quantity of action work we chose a large neural network."
                "This needs about 400 episodes to get a good policy, but unfortunately is quite volatile.")


def load_dqn_policy():
    """
    Load pretrained DQN policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleSwingShort-v0` env.

    :return: function pi: s -> a
    """
    import torch
    import numpy as np
    from challenge2.supplementary import DQN

    min_vol = -25  # Minimal voltage to apply
    max_vol = 25  # Maximal voltage to apply
    nn_outputs = 20  # Number of NeuralNetwork outputs, has to be > 1
    action_space = np.arange(
        min_vol, max_vol + 1,
                 (max_vol - min_vol) / (nn_outputs - 1),
        dtype=float
    )  # Discrete ActionSpace

    policy_nn = DQN(outputs=nn_outputs)  # Initialise Policy_Network

    checkpoint = torch.load("supplementary/DQN_checkpoint_DQN_checkpoint_massiv_nn_20VOLT_ACTUALLY10k_steps_20outputs_98gamma27Januar_16k.pt")
    policy_nn.load_state_dict(checkpoint['model_state_dict'])

    def get_action(obs):
        _, pos = torch.max(policy_nn(torch.Tensor(obs)), 0)  # Get action by policy
        # print('=======' + str(pos.item()) + '=====================')
        act = pos.item()
        return np.array([action_space[act]])

    return get_action


def train_dqn_policy(env):
    """
    Execute your implementation of the DQN learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """
    from challenge2 import DQNAlgorithmTraining

    dqn_training = DQNAlgorithmTraining(env)
    dqn_training.main()
    return dqn_training.get_policy_action


def load_lspi_policy():
    """
    Load pretrained LSPI policy from file.

    The policy must return a continuous action `a`
    that can be directly passed to `CartpoleStabShort-v0` env.

    :return: function pi: s -> a
    """

    from challenge2.supplementary.policy import PolicyTrained

    policy = PolicyTrained()
    return policy.best_act

def train_lspi_policy(env):
    """
    Execute your implementation of the LSPI learning algorithm.

    This function should start your code placed in a separate file.

    :param env: gym.Env
    :return: function pi: s -> a
    """
    from challenge2.lspi.lspi import LSPI
    lspi = LSPI(env)
    lspi.main()
    return lspi.changing_policy.best_act_np



# ==== Example evaluation
def main():
    import gym
    from gym.wrappers.monitor import Monitor
    import quanser_robots

    def evaluate(env, policy, num_evlas=25):
        ep_returns = []
        for eval_num in range(num_evlas):
            episode_return = 0
            dones = False
            obs = env.reset()
            while not dones:
                action = policy(obs)
                obs, rewards, dones, info = env.step(action)
                episode_return += rewards
            ep_returns.append(episode_return)
        return ep_returns

    def render(env, policy):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            act = policy(obs)
            obs, _, done, _ = env.step(act)

    def check(env, policy):
        render(env, policy)
        ret_all = evaluate(env, policy)
        print(np.mean(ret_all), np.std(ret_all))
        env.close()

    # DQN I: Check learned policy
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_eval')
    policy = load_dqn_policy()
    check(env, policy)

    # DQN II: Check learning procedure
    env = Monitor(gym.make('CartpoleSwingShort-v0'), 'dqn_train', video_callable=False)
    policy = train_dqn_policy(env)
    check(env, policy)

    # LSPI I: Check learned policy
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_eval')
    policy = load_lspi_policy()
    check(env, policy)

    # LSPI II: Check learning procedure
    env = Monitor(gym.make('CartpoleStabShort-v0'), 'lspi_train', video_callable=False)
    policy = train_lspi_policy(env)
    check(env, policy)


if __name__ == '__main__':
    main()
