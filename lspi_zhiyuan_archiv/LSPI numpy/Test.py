import gym
from gym.wrappers.monitor import Monitor
from quanser_robots.cartpole import ctrl
from quanser_robots.common import Logger, GentlyTerminating

import numpy as np
from Memory import Memory
from LSPI import LSPI

# Choose enviroment:
# 1 = CartPoleStab
# 2 = CartPoleSwingUp
# 3 = FurutaPend
env_choose = 1
if env_choose == 1:
    # env = Logger(GentlyTerminating(gym.make('CartpoleStabShort-v0')))
    env = gym.make('CartpoleStabShort-v0')

elif env_choose == 4:
    env = gym.make('CartpoleStabRR-v0')

elif env_choose == 2:
    env = gym.make('CartpoleSwingShort-v0')

else:
    env = gym.make('Qube-v0')

lspi = LSPI(env)


