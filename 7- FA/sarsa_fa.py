import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd


def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class SARSA:
    def __init__(self, state_dim, action_dim, gamma):
        self._q = Q(state_dim, action_dim)
        self._q.cuda()
        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0005)
        self._action_dim = action_dim

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes, time_steps, epsilon):

        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = env.reset()
            a = self.get_action(s, epsilon)
            for t in range(time_steps):
                ns, r, d, _ = env.step(a)


                na = self.get_action(ns, epsilon)

                target = r + (1 - d) * self._gamma * self._q(tt(ns))[na]
                current_prediction = self._q(tt(s))[a]

                loss = self._loss_function(current_prediction, target.detach())

                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()

                if d:
                    break

                s = ns
                a = na

