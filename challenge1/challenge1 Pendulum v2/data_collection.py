import numpy as np

import gym
import quanser_robots
from quanser_robots import GentlyTerminating
import time

#MagLev, CartPoleSwingUp, FurutaPend
#MagLev=Levitation-v0 ???
#CartPoleSwingUp=CartpoleSwingShort-v0
#FuturaPend=Qube-v0

env_names={0:"Levitation-v0",1:"CartpoleSwingShort-v0",2:"Qube-v0",3:"Pendulum-v2"}


ENV_NAME=env_names[3]
sampling_type="uniform"

print("Sampling env:")
print(ENV_NAME)

env=GentlyTerminating(gym.make(ENV_NAME))
print("Observation space:")
print(env.observation_space)
print("Low:")
print(env.observation_space.low)
print("High:")
print(env.observation_space.high)
print("Action space:")
print(env.action_space)
print("Low:")
print(env.action_space.low)
print("High:")
print(env.action_space.high)

states=[]
actions=[]
rewards=[]
next_states=[]

num_samples=50000
do_render=False

range=(env.action_space.high-env.action_space.low)[0]
print("Action Range:")
print(range)

assert(env.action_space.low.shape==(1,))

def random_action(mu,sigma):
  if sampling_type=="uniform":
    a=np.random.uniform(env.action_space.low[0],env.action_space.high[0],size=(1,))
  elif sampling_type=="discrete":
    a=np.random.choice([env.action_space.low[0],0,env.action_space.high[0]]).reshape(-1)
  else:
    a=np.random.normal(mu,sigma,size=(1,))
    a=np.clip(a,env.action_space.low,env.action_space.high)
  return a

while len(states)<num_samples:
  s=env.reset()
  #sample initial action uniformly from the action space
  mu=np.random.uniform(env.action_space.low[0],env.action_space.high[0])
  #exponential sampling for sigma of our markov chain of random actions
  sigma=np.exp(np.random.uniform(0,np.log(range*4)))
  done=False
  step=0
  while not done:
    if do_render:
        env.render()
        time.sleep(0.016)
    #do a step
    if step>-1:
        a=random_action(mu,sigma)
    else:
        #Bring Pendulum to bottom stand still.
        a=np.clip(-1*s[-1:],env.action_space.low[0],env.action_space.high[0])
    s_,r,done,info=env.step(a)
    #record data
    if not done:
      states.append(s)
      actions.append(a)
      rewards.append(r)
      next_states.append(s_)
      
    #update our Markov chain
    mu=a[0]
    #update current state
    del s
    s=s_
    step+=1


states=np.array(states)
actions=np.array(actions)
rewards=np.array(rewards)
next_states=np.array(next_states)

print("Observations Min:")
print(np.min(states,axis=0,keepdims=True))
print("Observations Max:")
print(np.max(states,axis=0,keepdims=True))
print("Observations Mean:")
print(np.mean(states,axis=0,keepdims=True))
print("Observations Std:")
print(np.std(states,axis=0,keepdims=True))

print("Actions Min:")
print(np.min(actions,axis=0))
print("Actions Max:")
print(np.max(actions,axis=0))
print("Actions Mean:")
print(np.mean(actions,axis=0))
print("Actions Std:")
print(np.std(actions,axis=0))

print("Rewards Min:")
print(np.min(rewards))
print("Rewards Max:")
print(np.max(rewards))
print("Rewards Mean:")
print(np.mean(rewards))
print("Rewards Std:")
print(np.std(rewards))

print(states.shape)
print(actions.shape)
print(rewards.shape)
print(next_states.shape)

np.save(f"dynamics_samples/{ENV_NAME}_states.npy",states)
np.save(f"dynamics_samples/{ENV_NAME}_actions.npy",actions)
np.save(f"dynamics_samples/{ENV_NAME}_rewards.npy",rewards)
np.save(f"dynamics_samples/{ENV_NAME}_next_states.npy",next_states)
