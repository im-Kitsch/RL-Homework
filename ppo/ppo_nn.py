import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import optim as optim

class PPONN(nn.Module):

    def __init__(self, num_input):
        super(PPONN, self).__init__()
        self.n_in = int(num_input)
        self.hid_layer1 = 60
        self.hid_layer2 = 60
        self.n_out = 1

        self.lin1 = nn.Linear(self.n_in, self.hid_layer1)
        self.lin2 = nn.Linear(self.hid_layer1, self.hid_layer2)

        self.head = nn.Linear(self.hid_layer2, self.n_out)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x)

    def minimize_loss(self, old_state, new_state, rwd):
        state_value = ValueNN.forward(old_state)
        # A = - state_value +

        # y_i = rwd + self.forward(new_state)
        # loss = self.loss_func(y_i, self.forward(old_state))
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return

class ValueNN(nn.Module):

    def __init__(self, num_input):
        super(ValueNN, self).__init__()
        self.n_in = num_input
        self.hid_layer1 = 60
        self.hid_layer2 = 60
        self.n_out = 1

        self.lin1 = nn.Linear(self.n_in, self.hid_layer1)
        self.lin2 = nn.Linear(self.hid_layer1, self.hid_layer2)

        self.head = nn.Linear(self.hid_layer2, self.n_out)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x)

    def minimize_loss(self, old_state, new_state, rwd):
        y_i = rwd + self.forward(new_state)
        loss = self.loss_func(y_i, self.forward(old_state))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()