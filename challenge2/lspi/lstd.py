import torch
import numpy as np


class LSTDQ:

    def __init__(self, gamma, obs_space_size):
        self.obs_space_size = obs_space_size
        self.gamma = gamma

    def learn(self, samples, policy):
        k = policy.bas_fun.size()
        B = torch.eye(k)*0.1
        b = torch.zeros((k, 1))

        for i in range(0, samples.shape[0]):
            sample = samples[i, :]
            obs = sample[0:self.obs_space_size]
            act = sample[self.obs_space_size]
            rew = sample[self.obs_space_size + 1].item()
            done = sample[sample.shape[0]-1]
            new_obs = sample[self.obs_space_size + 2:self.obs_space_size + self.obs_space_size + 2]

            # --------
            # Calc A
            # --------
            # phi(s, a)
            phi_obs_act = policy.bas_fun.calc_by_act(obs, act.item()).view(-1, 1)

            # pi(s')
            best_act = policy.best_act(new_obs)

            # phi(s', pi(s'))
            phi_new_obs_act = policy.bas_fun.calc_by_act(new_obs, best_act.item()).view(-1, 1)

            # phi(s,a) - gamma * phi(s, pi(s))^T
            h1 = torch.transpose((phi_obs_act - self.gamma * phi_new_obs_act), dim0=0, dim1=1)

            # B + phi(s,a)(phi(s,a)-gamma * phi(s', pi(s')))^T * B
            zaehler = torch.mm(B, torch.mm(phi_obs_act, torch.mm(h1, B)))

            # 1 + (phi(s,a)-gamma * phi(s',pi(s')))^T * B * phi(s,a)
            nenner = (1+torch.mm(h1, torch.mm(B, phi_obs_act)))
            B = B - zaehler/nenner

            b += torch.mul(phi_obs_act, rew)
        return torch.mm(B, b)
