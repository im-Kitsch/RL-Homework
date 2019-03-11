import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, inputs=5, outputs=9):
        super(DQN, self).__init__()
        # Size of Neural Network
        self.n_in  = int(inputs)
        self.n_hid1 = min(60, 60)
        self.n_hid2 = min(60, int(outputs*10.0))
        self.n_out = int(outputs)

        # weights
        self.lin1 = nn.Linear(self.n_in, self.n_hid1)
        # self.bn1 = nn.BatchNorm1d(60)
        self.lin2 = nn.Linear(self.n_hid1, self.n_hid2)
        self.lin3 = nn.Linear(self.n_hid1, self.n_hid2)
        self.lin4 = nn.Linear(self.n_hid1, self.n_hid2)
        self.lin5 = nn.Linear(self.n_hid1, self.n_hid2)
        # self.bn2 = nn.BatchNorm1d(min(60, int(outputs*10.0)))
        self.head = nn.Linear(self.n_hid2, self.n_out)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        return self.head(x)
