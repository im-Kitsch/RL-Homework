import torch as torch
import numpy as np
import matplotlib.pyplot as plt


class RadialBasisFunction:
    def __init__(self, beta, num_actions, obs_space_size, number_means, obs_space, stab):
        self.stab = stab
        self.obs_space = obs_space
        self.number_actions = num_actions
        self.beta = beta
        self.obs_space_size = obs_space_size
        self.number_means = number_means
        self.means = torch.zeros(self.obs_space_size, pow(self.number_means, self.obs_space_size))

    def set_beta(self, sigma):
        if self.stab:
            man_step = torch.tensor([128., 16., 8., 2.])
        else:
            man_step = torch.tensor([128., 16., 16., 8., 2.])
        self.beta = 1/(sigma) * man_step
        self.beta_n = torch.norm(self.beta)
        print("Beta: " + str(self.beta))

    def set_means(self, means, no_grid):
        if self.stab:
            self.no_grid = no_grid
            mesh_1, mesh_2, mesh_3, mesh_4 = torch.meshgrid(
                (no_grid[0, :], no_grid[1, :], no_grid[2, :], no_grid[3, :]))
            mesh_1 = mesh_1.contiguous().view(1, -1)
            mesh_2 = mesh_2.contiguous().view(1, -1)
            mesh_3 = mesh_3.contiguous().view(1, -1)
            mesh_4 = mesh_4.contiguous().view(1, -1)
            self.means = torch.cat((mesh_1, mesh_2, mesh_3, mesh_4), dim=0)
        else:
            self.no_grid = no_grid
            mesh_1, mesh_2, mesh_3, mesh_4, mesh_5 = torch.meshgrid(
                (no_grid[0, :], no_grid[1, :], no_grid[2, :], no_grid[3, :], no_grid[4, :]))
            mesh_1 = mesh_1.contiguous().view(1, -1)
            mesh_2 = mesh_2.contiguous().view(1, -1)
            mesh_3 = mesh_3.contiguous().view(1, -1)
            mesh_4 = mesh_4.contiguous().view(1, -1)
            mesh_5 = mesh_5.contiguous().view(1, -1)
            self.means = torch.cat((mesh_1, mesh_2, mesh_3, mesh_4, mesh_5), dim=0)
        # self.plot()

    def calc_by_act(self, obs, act):
        phi = torch.zeros(int(pow(self.number_means, self.obs_space_size) + 1) * self.number_actions)
        phi[int(act)*(self.means.shape[1]+1):int(act+1)*(self.means.shape[1]+1)] = torch.reshape(self.eval_obs(obs), (1, -1))
        return phi

    def size(self):
        return (self.means.shape[1] + 1) * self.number_actions

    def eval_obs(self, sample):
        t_sample = sample.view(-1, 1)
        d = torch.norm(t_sample-self.means, dim=0)
        v = torch.exp(-d/2)
        return torch.cat((torch.Tensor([1]).view(-1, 1), v.view(-1, 1)))

    def plot_eval(self, obs):
        diff = (obs.view(-1,1) - self.no_grid)
        return torch.exp(- torch.mul((diff * diff), self.beta.view(-1, 1)))

    def plot_eval_2(self, obs, i_1, i_2):
        dim_1 = self.no_grid[i_1, :]
        dim_2 = self.no_grid[i_2, :]

        need_1 = obs[i_1].view(-1, 1)
        need_2 = obs[i_2].view(-1, 1)
        obs_need = torch.cat((need_1, need_2))

        grid_x, grid_y = torch.meshgrid((dim_1, dim_2))
        grid_x = grid_x.contiguous().view(1, -1)
        grid_y = grid_y.contiguous().view(1, -1)
        t_grid = torch.cat((grid_x, grid_y), dim=0)
        dist = torch.norm(obs_need-t_grid, dim=0)
        e = torch.exp(-dist/4)
        return torch.exp(-dist/4)




    def plot(self):
        test = []
        steps = 400

        t_x = torch.reshape(torch.linspace(self.obs_space.low[0].item(), self.obs_space.high[0].item(), steps=steps),
                            (-1, 1))
        t_s = torch.reshape(torch.linspace(self.obs_space.low[1].item(), self.obs_space.high[1].item(), steps=steps),
                            (-1, 1))
        t_c = torch.reshape(torch.linspace(self.obs_space.low[2].item(), self.obs_space.high[2].item(), steps=steps),
                            (-1, 1))
        t_p_d = torch.reshape(torch.linspace(-1.5, 1.5, steps=steps), (-1, 1))
        t_x_d = torch.reshape(torch.linspace(-10, 10, steps=steps), (-1, 1))

        test.append(t_x.view(1, -1))
        test.append(t_s.view(1, -1))
        test.append(t_c.view(1, -1))
        test.append(t_p_d.view(1, -1))
        test.append(t_x_d.view(1, -1))
        test_t = torch.cat(test)

        y_x = []
        y_s = []
        y_c = []
        y_p_d = []
        y_x_d = []
        for i in range(0, test_t.shape[1]):
            h = self.plot_eval(test_t[:, i])
            y_x.append(h[0, :].view(1, -1))
            y_s.append(h[1, :].view(1, -1))
            y_c.append(h[2, :].view(1, -1))
            y_p_d.append(h[3, :].view(1, -1))
            y_x_d.append(h[4, :].view(1, -1))

        y_x = torch.cat(y_x)
        y_s = torch.cat(y_s)
        y_c = torch.cat(y_c)
        y_p_d = torch.cat(y_p_d)
        y_x_d = torch.cat(y_x_d)

        f, ax = plt.subplots(self.obs_space.shape[0], 1)
        f.subplots_adjust(hspace=1)

        for i in range(0, self.obs_space_size):
            axi = ax[i]
            for j in range(0, self.number_means):
                if i == 0:
                    axi.plot(t_x.detach().numpy(), y_x[:, j].detach().numpy())
                    axi.set_xlim(-0.1, 0.1)
                    axi.set_xlabel('x')
                elif i == 1:
                    axi.plot(t_s.detach().numpy(), y_s[:, j].detach().numpy())
                    axi.set_xlim(-0.5, 0.5)
                    axi.set_xlabel('sin(phi)')
                elif i == 2:
                    axi.plot(t_c.detach().numpy(), y_c[:, j].detach().numpy())
                    axi.set_xlabel('cos(phi)')
                elif i == 3:
                    axi.plot(t_p_d.detach().numpy(), y_p_d[:, j].detach().numpy())
                    axi.set_xlabel('velocity(x)')
                elif i == 4:
                    axi.plot(t_x_d.detach().numpy(), y_x_d[:, j].detach().numpy())
                    axi.set_xlabel('velocity(phi)')

        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig('rbf_functions')

        dist = []
        for i in range(0, test_t.shape[1]):
            h2 = self.plot_eval_2(test_t[:, i], 4, 3)
            dist.append(h2)

        dist = torch.stack(dist)

        plt.figure(5)
        for i in range(0, dist.shape[1]):
            plt.plot(t_x.detach().numpy(), dist[:, i].detach().numpy())

        plt.show(block=False)
        plt.pause(0.01)
        print('Plotting Finished')

        '''
        plt.figure(5)
        t = np.arange(0, steps, 1)
        plt.plot(t_x.numpy(), y.numpy())

        # plt.show()
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig('rbf_functions')
        '''


        '''
        t_x = torch.reshape(torch.arange(self.obs_space.low[0].item(), self.obs_space.high[0].item(), (self.obs_space.high[0].item()-self.obs_space.low[0].item())/steps), (-1, 1))
        t_s = torch.reshape(torch.arange(self.obs_space.low[1].item(), self.obs_space.high[1].item(), (self.obs_space.high[1].item()-self.obs_space.low[1].item())/steps), (-1, 1))
        t_p_d = torch.reshape(torch.arange(-1.5, 1.5, (1.5 + 1.5)/steps), (-1, 1))
        t_x_d = torch.reshape(torch.arange(-10, 10, (10 + 10)/steps), (-1, 1))

        test.append(t_x.view(1, -1))
        test.append(t_s.view(1, -1))
        if not self.stab:
            t_c = torch.reshape(torch.arange(self.obs_space.low[2].item(), self.obs_space.high[2].item(), (self.obs_space.high[2].item()-self.obs_space.low[2].item())/steps), (-1, 1))
            test.append(t_c.view(1, -1))
        test.append(t_p_d.view(1, -1))
        test.append(t_x_d.view(1, -1))
        test_t = torch.cat(test)

        y = []
        for i in range(0, test_t.shape[1]):
            y.append(torch.transpose(self.plot_calc(test_t[:, i]), 0, 1))

        y = torch.cat(y)

        f, ax = plt.subplots(self.obs_space.shape[0], 1)
        f.subplots_adjust(hspace=1)

        for i in range(0, self.obs_space_size):
            axi = ax[i]
            for j in range(0, self.number_means):
                axi.plot(test_t[i, :].detach().numpy(), y[:, i+self.obs_space_size*j].detach().numpy())
            if i == 0:
                axi.set_xlim(-0.1, 0.1)
                axi.set_xlabel('x')
            elif i == 1:
                axi.set_xlim(-0.5, 0.5)
                axi.set_xlabel('sin(phi)')
            elif i == 2:
                axi.set_xlabel('cos(phi)')
            elif i == 3:
                axi.set_xlabel('velocity(x)')
            elif i == 4:
                axi.set_xlabel('velocity(phi)')

        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig('rbf_functions')
        '''

class Polynomial:
    def __init__(self, degree, obs_size):
        self.obs_size = obs_size
        self.degree = degree

    def calc_by_act(self, obs, act):
        phi = torch.zeros(int(pow(self.number_means, self.obs_space_size) + 1) * self.number_actions)
        phi[int(act)*(self.means.shape[1]+1):int(act+1)*(self.means.shape[1]+1)] = torch.reshape(self.eval_obs(obs), (1, -1))
        return phi

    def eval_obs(self):
        return 0