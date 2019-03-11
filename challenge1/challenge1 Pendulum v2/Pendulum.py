#!/usr/bin/env python
# coding: utf-8

# In[24]:


import torch
import numpy as np
import gym
import quanser_robots
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

st_quan = [250,100]
a_quan = [3]

s_len = 2
a_len = 1

## old discretation function
# def state_cont2dis(conti, possible_dis):
#     if conti.ndim == 1 : #single state point
#         conti = conti.reshape(1,-1)
#     assert possible_dis.ndim == 2
#     assert conti.shape[1] == possible_dis.shape[1]
    
#     possible_dis = possible_dis.reshape( 1, possible_dis.shape[0], possible_dis.shape[1])
    
#     conti = conti.reshape(conti.shape[0], 1, conti.shape[1])
    
#     err = np.abs(possible_dis-conti)
#     err = np.sum(err, axis=-1)
    
#     index = np.argmin(err, axis=-1)
#     del err
#     possible_dis = possible_dis.reshape( -1, possible_dis.shape[2])
#     return index, possible_dis[index]

def state_cont2dis(conti, s1_dis, s2_dis, possible_combi):
    if conti.ndim == 1 : #single state point
        conti = conti.reshape(1,-1)
    assert possible_combi.ndim == 2
    assert conti.shape[1] == possible_combi.shape[1]
    
    s1_dis = s1_dis.reshape(1,-1)
    s2_dis = s2_dis.reshape(1,-1)
    s1 = conti[:,0].reshape(-1,1)
    s2 = conti[:,1].reshape(-1,1)
    err1 = np.abs(s1 - s1_dis)
    err2 = np.abs(s2 - s2_dis)
  
    index1 = np.argmin(err1,axis=-1)
    index2 = np.argmin(err2,axis=-1)
    index = index1*s2_dis.shape[1] + index2
    states = possible_combi[index]
    return index, states


reward_model = torch.load(f"dynamics_models/Pendulum_rewards.pth")
state_model= torch.load(f"dynamics_models/Pendulum_states.pth")
reward_model.eval()
state_model.eval()

para = np.load(f"dynamics_models/Data_para.npz")
# print(para.files)
st_mean = para["st_mean"].reshape(1,-1)
st_std = para["st_std"].reshape(1,-1)
r_mean = para["r_mean"]
r_std = para["r_std"]
a_mean = para["a_mean"]
a_std = para["a_std"]

env = gym.make("Pendulum-v2")
a_low, a_high = env.action_space.low, env.action_space.high
s_low, s_high = env.observation_space.low, env.observation_space.high

#-------------
if True: #if not manualy set discretization
    dis_actions = np.linspace(a_low, a_high, a_quan[0], dtype=float)
    dis_states1 = np.linspace(s_low[0], s_high[0], st_quan[0], dtype=float)
    dis_states2 = np.linspace(s_low[1], s_high[1], st_quan[1], dtype=float)
    
#----------------------------------------------
# old state discretezation
# A, B = np.meshgrid(dis_states1, dis_states2)
# possible_states = np.concatenate( [A.reshape(-1,1), B.reshape(-1,1)], axis=-1)
# del A,B,dis_actions

#----------------------------------------
# new state discretezation
possible_states = np.zeros((len(dis_states1), len(dis_states2), s_len ) )
possible_states[:,:,0] = dis_states1.reshape(-1, 1)# Becareful for the shape!!!
possible_states[:,:,1] = dis_states2.reshape( 1, -1)  
possible_states = possible_states.reshape(-1,2)

possible_actions = dis_actions

s_a_pair = np.zeros((len(possible_states), len(possible_actions), s_len + a_len ))
s_a_pair[:, :, :s_len] = possible_states.reshape(-1,1,s_len)
s_a_pair[:, :, s_len:] = possible_actions.reshape(1,-1,a_len)

print("created state action pair: ", s_a_pair.shape)


# get s'(s,a) and r_s_a

inputs = s_a_pair.reshape(-1, s_len+a_len)
del s_a_pair

inputs = np.concatenate( 
    [np.sin(inputs[:,0]).reshape(-1,1), np.cos(inputs[:,0]).reshape(-1,1), inputs[:,1:]], 
    axis = -1)
inputs[:,:3] = (inputs[:,:3] - st_mean)/st_std
inputs[:,3:] = ((inputs[:,3:] - a_mean))/a_std
inputs = torch.tensor(inputs)

r_s_a = reward_model(inputs)
r_s_a = r_s_a.detach().numpy()
r_s_a = r_s_a*r_std + r_mean
r_s_a = r_s_a.reshape(len(possible_states), len(possible_actions))

sne_s_a = state_model(inputs)
sne_s_a = sne_s_a.detach().numpy()
sne_s_a = sne_s_a*st_std + st_mean
sne_s_a = np.stack(
    [ np.arctan2(sne_s_a[:,0],sne_s_a[:,1]), sne_s_a[:,2] ],
    axis=-1)
del inputs, state_model, reward_model

print("finished combining all state action reward nextsate")

#important is index, state not used by the iteration
ne_index, dis_st = state_cont2dis(sne_s_a, dis_states1, dis_states2, possible_states) 
dis_err = np.sum(  np.abs( dis_st- sne_s_a ) )

print("For %d states"%len(possible_states), "discretization error: ", dis_err )

ne_index = ne_index.reshape(len(possible_states), len(possible_actions))


gamma = 1 - 1e-3
v = np.zeros((possible_states.shape[0], 1))
V = np.zeros_like(v)
q = np.zeros_like(r_s_a)

t = 0
while t<50000:
    t+=1
    
    v = V
    q = r_s_a + gamma * V[ne_index].reshape(ne_index.shape)
    assert q.ndim==2
    assert q.shape[0] == len(possible_states)
    assert q.shape[1] == len(possible_actions)
    V = np.max(q, axis=-1)
    assert V.shape[0] == possible_states.shape[0]
    
    err = np.abs(v-V)
    if np.max(err) <1e-7:
        print("succeseful")
        break
    print(t, np.max(err))
policy = possible_actions[np.argmax(q,axis=1)]


# In[29]:


import time

do_render = False
episodes = 100

s_record = []
a_record = []
r_record = []
sne_record = []

for i in range(episodes):
    
    s=env.reset()
    done = False
    
    s_r = []
    a_r = []
    r_r = []
    sne_r = []
   
    while not done:        
        if do_render:
            env.render()
            time.sleep(0.016)
        s=np.array(s).reshape(1,-1)
        s_idx,_= state_cont2dis(s, dis_states1, dis_states2, possible_states)
        a=policy[s_idx]
        
        s_r.append(s)
        a_r.append(a)           
        
        s,r,done,info=env.step(a)
        
        sne_r.append(s)
        r_r.append(r)        
    s_record.append(s_r)
    a_record.append(a_r)
    sne_record.append(sne_r)
    r_record.append(r_r)
env.close()

r_record = np.array(r_record)
assert r_record.shape[0] == episodes
cumu_r = np.sum(r_record, axis=1)


plt.plot(cumu_r)
plt.plot( [np.mean(cumu_r)]*episodes )
plt.plot( [-500]*episodes )

plt.show()
print(np.count_nonzero(cumu_r>-500.))

np.save("v2policy.npy", policy)

