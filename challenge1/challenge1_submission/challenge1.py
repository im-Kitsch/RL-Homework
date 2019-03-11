"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""

import time
import torch
import numpy as np
import torch.utils.data as Data
import torch.optim as optim


info = dict(
    group_number=4,  # change if you are an existing seminar/project group
    authors="Zhiyuan Hu; Christopher Sura; Alexander Gübler",
    description="""
        pipeline: collect data -> neural network train model -> value iteration
        collect data,  a_t = a_(t-1) + Gaussian Noise
        training model: Normalizing the data in 0:1,
        value iteration: important is that the discretization is based on theta and theta_d rather than [cos sin theta]
        """)


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """

    def random_action(mu, sigma):
        if sampling_type == "uniform":
            a = np.random.uniform(env.action_space.low[0], env.action_space.high[0], size=(1,))
        elif sampling_type == "discrete":
            a = np.random.choice([env.action_space.low[0], 0, env.action_space.high[0]]).reshape(-1)
        else:
            a = np.random.normal(mu, sigma, size=(1,))
            a = np.clip(a, env.action_space.low, env.action_space.high)
        return a

    sampling_type = "discrete"

    print("Observation space:", env.observation_space, "Low:", env.observation_space.low, "High:",
          env.observation_space.high)

    print("Action space:", env.action_space, "Low:", env.action_space.low, "High:", env.action_space.high)

    states = []
    actions = []
    rewards = []
    next_states = []

    num_samples = max_num_samples
    do_render = False

    arange = (env.action_space.high - env.action_space.low)[0]
    print("Action Range:", arange)

    assert (env.action_space.low.shape == (1,))

    # Although the number of samples are little bit more than max_num_samples, but only max_num_samples is used
    while len(states) < num_samples:
        s = env.reset()
        # sample initial action uniformly from the action space
        mu = np.random.uniform(env.action_space.low[0], env.action_space.high[0])
        # exponential sampling for sigma of our markov chain of random actions
        sigma = np.exp(np.random.uniform(0, np.log(arange * 4)))
        done = False
        step = 0
        while not done:
            if do_render:
                env.render()
                time.sleep(0.016)
            # do a step
            if step > -1:
                a = random_action(mu, sigma)
            else:
                # Bring Pendulum to bottom stand still.
                a = np.clip(-1 * s[-1:], env.action_space.low[0], env.action_space.high[0])
            s_, r, done, info = env.step(a)
            # record data
            if not done:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(s_)

            # update our Markov chain
            mu = a[0]
            # update current state
            del s
            s = s_
            step += 1

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)

    print("Observations Min:", np.min(states, axis=0, keepdims=True))
    print("Observations Max:", np.max(states, axis=0, keepdims=True))
    print("Observations Mean:", np.mean(states, axis=0, keepdims=True))
    print("Observations Std:", np.std(states, axis=0, keepdims=True))

    print("Actions Min:", np.min(actions, axis=0))
    print("Actions Max:", np.max(actions, axis=0))
    print("Actions Mean:", np.mean(actions, axis=0))
    print("Actions Std:", np.std(actions, axis=0))

    print("Rewards Min:", np.min(rewards))
    print("Rewards Max:", np.max(rewards))
    print("Rewards Mean:", np.mean(rewards))
    print("Rewards Std:", np.std(rewards))

    # np.savez(f"{ENV_NAME}_samples.npz", states=states, actions=actions, rewards=rewards, next_states=next_states)

    # from collections import OrderedDict

    traing_size = int(max_num_samples * 0.85)
    validation_size = max_num_samples - traing_size

    st_mean = states.min(axis=0)
    st_std = np.max(states - st_mean, axis=0)
    a_mean = -2
    a_std = 4
    r_mean = -20  # np.min(rewards)
    r_std = 20  # np.max(rewards-r_mean)

    states = (states - st_mean) / st_std
    next_states = (next_states - st_mean) / st_std
    rewards = (rewards - r_mean) / r_std
    actions = (actions - a_mean) / a_std

    # np.saves("dynamics_models/Data_para.npz", st_mean=st_mean, st_std=st_std, r_mean=r_mean, r_std=r_std,
    #          a_mean=a_mean, a_std=a_std)

    random_index = np.random.permutation(traing_size + validation_size)
    traing_index = random_index[:traing_size]
    testing_index = random_index[traing_size:traing_size + validation_size]

    test_states, test_actions, test_rewards, test_next_states = states[traing_index], actions[traing_index], rewards[
        traing_index], next_states[traing_index]

    states, actions, rewards, next_states = states[traing_index], actions[traing_index], rewards[traing_index], \
                                            next_states[traing_index]

    epochs = 40
    batch_size = 32

    inputs, outputs = np.concatenate((states, actions), axis=1), rewards
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    test_inputs, test_outputs = np.concatenate((test_states, test_actions), axis=1), test_rewards
    test_inputs, test_outputs = torch.tensor(test_inputs), torch.tensor(test_outputs)

    reward_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        #             torch.nn.Linear(64, 64),
        #             torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).double()  # Attention the double here!!
    print("NN Model")
    print(reward_model)

    dataset = Data.TensorDataset(inputs, outputs)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    criterion = torch.nn.MSELoss()
    optimizer = optim.RMSprop(reward_model.parameters(), lr=0.001, centered=True)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

    for epoch in range(epochs):
        epoch_loss = 0.0
        lr_scheduler.step()
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_y = batch_y.reshape(-1, 1)
            batch_y_p = reward_model(batch_x)
            loss = criterion(batch_y_p, batch_y)

            epoch_loss += batch_y.shape[0] * loss.item()
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        vali_loss = criterion(reward_model(test_inputs), test_outputs.reshape(-1, 1)).item()
        print("epoch: ", epoch + 1, "traing_loss: %e" % (epoch_loss / traing_size), "validation_loss: %e" % vali_loss)

    # In[3]:

    # Train State Model

    epochs = 60
    batch_size = 64

    inputs, outputs = np.concatenate((states, actions), axis=1), next_states
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    test_inputs, test_outputs = np.concatenate((test_states, test_actions), axis=1), test_next_states
    test_inputs, test_outputs = torch.tensor(test_inputs), torch.tensor(test_outputs)

    state_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        #             torch.nn.Linear(64, 64),
        #             torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    ).double()  # Attention the double here!!
    print("NN Model")
    print(state_model)

    dataset = Data.TensorDataset(inputs, outputs)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    criterion = torch.nn.MSELoss()
    optimizer = optim.RMSprop(state_model.parameters(), lr=0.001, centered=True)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

    for epoch in range(epochs):
        lr_scheduler.step()
        epoch_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(loader):
            batch_y_p = state_model(batch_x)
            loss = criterion(batch_y_p, batch_y)

            epoch_loss += batch_y.shape[0] * loss.item()
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        vali_loss = criterion(state_model(test_inputs), test_outputs).item()
        print("epoch: ", epoch, "traing_loss: %e" % (epoch_loss / traing_size), "validation_loss: %e" % vali_loss)

    # In[5]:

    reward_model.eval()
    state_model.eval()
    # Save the model
    # torch.save(reward_model, f"dynamics_models/Pendulum_rewards.pth")
    # torch.save(state_model, f"dynamics_models/Pendulum_states.pth")

    def model(obs, act):
        obs = obs.reshape(-1, env.observation_space.low.shape[0])
        act = act.reshape(-1, 1)
        inputs = np.concatenate((obs, act), axis=1)

        inputs[:, :3] = (inputs[:, :3] - st_mean) / st_std
        inputs[:, 3:] = (inputs[:, 3:] - a_mean) / a_std
        inputs = torch.tensor(inputs)

        r_s_a = reward_model(inputs)
        r_s_a = r_s_a.detach().numpy()
        r_s_a = r_s_a * r_std + r_mean

        sne_s_a = state_model(inputs)
        sne_s_a = sne_s_a.detach().numpy()
        sne_s_a = sne_s_a * st_std + st_mean

        return sne_s_a, r_s_a

    return model  # lambda obs, act: (2*obs + act, obs@obs + act**2)


def state_cont2dis(conti, s1_dis, s2_dis, possible_combi):
    if conti.ndim == 1:  # single state point
        conti = conti.reshape(1, -1)
    assert possible_combi.ndim == 2
    assert conti.shape[1] == possible_combi.shape[1]

    s1_dis = s1_dis.reshape(1, -1)
    s2_dis = s2_dis.reshape(1, -1)
    s1 = conti[:, 0].reshape(-1, 1)
    s2 = conti[:, 1].reshape(-1, 1)
    err1 = np.abs(s1 - s1_dis)
    err2 = np.abs(s2 - s2_dis)

    index1 = np.argmin(err1, axis=-1)
    index2 = np.argmin(err2, axis=-1)
    index = index1 * s2_dis.shape[1] + index2
    states = possible_combi[index]
    return index, states


# def view_bar(num, mes, err):
#     old_num = num; num = num/mes*100; mes = 100
#     rate_num = num
#     number = int(rate_num / 4)
#     hashes = '=' * number
#     spaces = ' ' * (25 - number)
#     r = "\r\033[31;0m%s\033[0m：[%s%s]\033[32;0m%d%% \033[0m  " % ("ite, "+str(old_num),
#                                                                hashes, spaces, rate_num, )
#     sys.stdout.write(r); sys.stdout.flush()
#     return


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """

    st_quan = [150, 130]
    a_quan = [3]

    s_len = 2
    a_len = 1

    a_low, a_high = action_space.low, action_space.high
    s_low, s_high = observation_space.low, observation_space.high
    s_low, s_high = np.array([-np.pi, s_low[2]]), np.array([np.pi, s_high[2]])
    print(s_low)
    print(s_high)
    # -------------
    if True:  # if not manualy set discretization
        dis_actions = np.linspace(a_low, a_high, a_quan[0], dtype=float)
        dis_states1 = np.linspace(s_low[0], s_high[0], st_quan[0], dtype=float)
        dis_states2 = np.linspace(s_low[1], s_high[1], st_quan[1], dtype=float)

    # ----------------------------------------
    #  state discretezation
    possible_states = np.zeros((len(dis_states1), len(dis_states2), s_len))
    possible_states[:, :, 0] = dis_states1.reshape(-1, 1)  # Becareful for the shape!!!
    possible_states[:, :, 1] = dis_states2.reshape(1, -1)
    possible_states = possible_states.reshape(-1, 2)

    possible_actions = dis_actions

    s_a_pair = np.zeros((len(possible_states), len(possible_actions), s_len + a_len))
    s_a_pair[:, :, :s_len] = possible_states.reshape(-1, 1, s_len)
    s_a_pair[:, :, s_len:] = possible_actions.reshape(1, -1, a_len)

    print("created state action pair: ", s_a_pair.shape)

    # get s'(s,a) and r_s_a

    inputs = s_a_pair.reshape(-1, s_len + a_len)
    del s_a_pair

    inputs = np.concatenate(
        [np.cos(inputs[:, 0]).reshape(-1, 1), np.sin(inputs[:, 0]).reshape(-1, 1), inputs[:, 1:]],
        axis=-1)

    sne_s_a, r_s_a = model(inputs[:, :3], inputs[:, 3].reshape(-1, 1))

    sne_s_a = np.stack(
            [np.arctan2(sne_s_a[:, 1], sne_s_a[:, 0]), sne_s_a[:, 2]],
            axis=-1)

    print("finished combining all state action reward nextsate")

    # important is index, state not used by the iteration
    ne_index, dis_st = state_cont2dis(sne_s_a, dis_states1, dis_states2, possible_states)
    dis_err = np.sum(np.abs(dis_st - sne_s_a))

    print("For %d states" % len(possible_states), "discretization error: ", dis_err)

    del dis_err, dis_st, sne_s_a
    r_s_a = r_s_a.reshape(len(possible_states), len(possible_actions))
    ne_index = ne_index.reshape(len(possible_states), len(possible_actions))

    gamma = 1 - 1e-3
    v = np.zeros((possible_states.shape[0], 1))
    V = np.zeros_like(v)
    q = np.zeros_like(r_s_a)

    t = 0
    while t < 50000:

        t += 1
        v = V
        q = r_s_a + gamma * V[ne_index].reshape(ne_index.shape)
        assert q.ndim == 2
        assert q.shape[0] == len(possible_states)
        assert q.shape[1] == len(possible_actions)
        V = np.max(q, axis=-1)
        assert V.shape[0] == possible_states.shape[0]

        err = np.abs(v - V)
        print(t, np.max(err))
        if np.max(err) < 1e-7:
            print("value iteration succeseful")
            break

    policy = possible_actions[np.argmax(q, axis=1)]

    def policy_func(obs):
        obs = obs.reshape(-1, observation_space.low.shape[0])
        # ne_index, dis_st = state_cont2dis(sne_s_a, dis_states1, dis_states2, possible_states)
        s = obs
        angle = np.arctan2(s[:, 1], s[:, 0])

        s = np.concatenate((angle.reshape(-1, 1), s[:, 2].reshape(-1, 1)), axis=1)

        s_idx, _ = state_cont2dis(s, dis_states1, dis_states2, possible_states)
        a = policy[s_idx]

        return a

    return policy_func  # lambda obs: action_space.high
