#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
# from collections import OrderedDict

traing_size = 10000
validation_size = 2000



env_names={0:"Levitation-v0",1:"CartpoleSwingShort-v0",2:"Qube-v0",3:"Pendulum-v2"}
ENV_NAME=env_names[3]

states=np.load(f"dynamics_samples/{ENV_NAME}_states.npy")[:traing_size+validation_size]
actions=np.load(f"dynamics_samples/{ENV_NAME}_actions.npy")[:traing_size+validation_size]
rewards=np.load(f"dynamics_samples/{ENV_NAME}_rewards.npy")[:traing_size+validation_size].reshape(-1,1)
next_states=np.load(f"dynamics_samples/{ENV_NAME}_next_states.npy")[:traing_size+validation_size]

#----------------change the state---------------
states = np.concatenate([
    np.sin(states[:,0]).reshape(-1,1), np.cos(states[:,0]).reshape(-1,1), 
    states[:,1].reshape(-1,1)
], axis = 1)

next_states = np.concatenate([
    np.sin(next_states[:,0]).reshape(-1,1), np.cos(next_states[:,0]).reshape(-1,1), 
    next_states[:,1].reshape(-1,1)
], axis = 1)

#-----------Normalization
# Data Normalization
# st_mean = states.mean(axis=0)
# st_std = states.std(axis=0)
# a_mean = actions.mean()
# a_std = actions.std()
# r_mean = rewards.mean()
# r_std = states.std()

#Actually it's range
st_mean = states.min(axis=0)
st_std = np.max(states - st_mean, axis=0)
a_mean = -2
a_std = 4
r_mean = -20# np.min(rewards)
r_std = 20 #np.max(rewards-r_mean)

states = (states-st_mean)/(st_std)
next_states = (next_states-st_mean)/st_std
rewards = (rewards-r_mean)/r_std
actions = (actions-a_mean)/a_std

np.savez("dynamics_models/Data_para.npz", st_mean=st_mean, st_std=st_std, r_mean= r_mean, r_std=r_std,
         a_mean=a_mean, a_std=a_std)

random_index = np.random.permutation(traing_size+validation_size)
traing_index = random_index[:traing_size]
testing_index = random_index[traing_size:traing_size+validation_size]

test_states, test_actions, test_rewards, test_next_states =      states[traing_index], actions[traing_index],     rewards[traing_index], next_states[traing_index]

states, actions, rewards, next_states =     states[traing_index], actions[traing_index], rewards[traing_index], next_states[traing_index]


# In[2]:


# Train Reward Model

epochs = 40
batch_size = 32

inputs, outputs = np.concatenate((states,actions), axis = 1), rewards
inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
test_inputs, test_outputs = np.concatenate((test_states,test_actions), axis = 1), test_rewards
test_inputs, test_outputs = torch.tensor(test_inputs), torch.tensor(test_outputs)

reward_model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
).double()   # Attention the double here!!
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
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=(0.985))

for epoch in range(epochs):
    epoch_loss = 0.0
    lr_scheduler.step()
    for step, (batch_x, batch_y) in enumerate(loader):        
        batch_y_p = reward_model(batch_x)
        loss = criterion(batch_y_p, batch_y)
        
        epoch_loss += batch_y.shape[0] * loss.item()
        #Backward
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
    
    vali_loss = criterion(reward_model(test_inputs), test_outputs).item()
    print("epoch: ", epoch+1, "traing_loss: %e"%(epoch_loss/traing_size), "validation_loss: %e"%vali_loss) 


# In[3]:


# Train State Model

epochs = 60
batch_size = 64

inputs, outputs = np.concatenate((states,actions), axis = 1), next_states
inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
test_inputs, test_outputs = np.concatenate((test_states,test_actions), axis = 1), test_next_states
test_inputs, test_outputs = torch.tensor(test_inputs), torch.tensor(test_outputs)

state_model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
).double()   # Attention the double here!!
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
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=(0.985))

for epoch in range(epochs):
    lr_scheduler.step()
    epoch_loss = 0.0
    for step, (batch_x, batch_y) in enumerate(loader):        
        batch_y_p = state_model(batch_x)
        loss = criterion(batch_y_p, batch_y)
        
        epoch_loss += batch_y.shape[0] * loss.item()
        #Backward
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
    
    vali_loss = criterion(state_model(test_inputs), test_outputs).item()
    print("epoch: ", epoch, "traing_loss: %e"%(epoch_loss/traing_size), "validation_loss: %e"%vali_loss)       


# In[5]:


reward_model.eval()
state_model.eval()
# Save the model
torch.save(reward_model, f"dynamics_models/Pendulum_rewards.pth")
torch.save(state_model, f"dynamics_models/Pendulum_states.pth")

