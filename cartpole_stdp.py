# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 08:18:38 2021

@author: lenovo
"""

import gym
from typing import Optional, NamedTuple, Tuple, Any, Sequence
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(8) #7/8/13
# time constants (in ms)
TAU_V = 10.0  # voltage
V_DECAY = torch.exp(-1.0 / torch.tensor(TAU_V))
TAU_I = 2.0  # current
I_DECAY = torch.exp(-1.0 / torch.tensor(TAU_I))
TAU_O = 20.0  # neuron trace
O_DECAY = torch.exp(-1.0 / torch.tensor(TAU_O))
TAU_E = 100.0 # synapse trace (from paper)
E_DECAY = torch.exp(-1.0 / torch.tensor(TAU_E))
# thresh
THRESH = torch.tensor(1.0)

# inputs
N = 40    # neurons in spiking pop 50
T = 100   # length of single seq 500

# learning
EPOCHS = 100

def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return x.gt(0).float()

def lif(x: torch.Tensor, s: Optional[torch.Tensor]) -> Tuple[torch.Tensor]:
    if s is None:
        s = torch.zeros(4, *x.shape, device=x.device, dtype=x.dtype)
    # spikes, voltage, current, out trace
    z, v, i, o = s  # unbind, removes dimension
    
    i_new = i * I_DECAY + (1 - I_DECAY) * x
    v_new = v * V_DECAY * (1 - z) + (1 - V_DECAY) * i_new
    z_new = spike_fn(v_new - THRESH)
    o_new = o * O_DECAY - z_new  #*(1 - O_DECAY)
    return z_new, torch.stack([z_new, v_new, i_new, o_new])

def forward(x: torch.Tensor, s: Optional[torch.Tensor], param) -> Tuple[torch.Tensor, torch.Tensor]:
    # check dimensions
    assert x.dim() == 1
    assert param["W"].dim() == 2
    
    # multiply input spikes with weight
    x = param["W"] @ x
    
    # LIF
    z_new, s_new = lif(x, s)
    return z_new, s_new

#def encoding(states:torch.Tensor):
#    spikes = torch.randn(4, int(N/4))
#    states = states.unsqueeze(-1)
#    spike_train = (spikes < states).float()
#    spike_train = spike_train.reshape(N,)
#    return spike_train

def current_encoding(obs:torch.Tensor, param):   
    assert obs.dim() == 1
    assert param["W"].dim() == 2
    input_current = param["W"] @ obs
    i_new = (1 - I_DECAY) * input_current
    v_new = (1 - V_DECAY) * i_new
    z_new = spike_fn(v_new - 0) #input_spikes    
    return z_new

def place_cell(obs:torch.Tensor):
    spikes = torch.zeros((40*4,))
    spikes[int(obs[0].floor())+20] = 1
    spikes[int(obs[1].floor())+60] = 1
    spikes[int(obs[2].floor())+100] = 1
    spikes[int(obs[3].floor())+140] = 1
    return spikes

def get_reward(z: torch.Tensor, label: int) -> Tuple[int, int]:
    # return reward and correct for counter
    #If the correct output was 1,the network received a reward r = 1 for each output spike emitted and 0 otherwise
    #If the correct output was 0, the network received a negative reward (punishment) r = −1 for each output spike and 0 otherwise.
    if z.item() == 1:
        if label == 1:
            return 1 
        else:
            return 0 
    elif z.item() == 0:
        return 0  
    else:
        raise ValueError("y negative")

param = {}
param["W"] = torch.randn(1, N) * 10 * (torch.rand(1, N) < 0.8).float()
print(param)
param1 = {}
param1["W"] = torch.randn(N, 4) * 10 * (torch.rand(N, 4) < 0.8).float()
param2 = {}
param2["W"] = torch.randn(N, 8) * 10 * (torch.rand(N, 8) < 0.8).float()
#print(param1)
#test = torch.Tensor([1.0,-1.0,1.0,0])
#a = torch.Tensor([[1+torch.sign(test[0]),-1+torch.sign(test[0]),0,0,0,0,0,0],
#                 [0,0,1+torch.sign(test[1]),-1+torch.sign(test[1]),0,0,0,0],
#                 [0,0,0,0,1+torch.sign(test[2]),-1+torch.sign(test[2]),0,0],
#                [0,0,0,0,0,0,1+torch.sign(test[3]),-1+torch.sign(test[3])]])
#print(param1["W"] @ test)
#b = torch.Tensor([19.5,-19.5,1.0,19.5])
#c = torch.zeros((160,))
#c[int(b[0].floor())+20] = 1
#c[int(b[1].floor())+60] = 1
#c[int(b[2].floor())+100] = 1
#c[int(b[3].floor())+140] = 1
#print(c)

env = gym.make('CartPole-v0')  
env.reset()
ETA = 0.001
tot_reward_tab = []


for e in range(EPOCHS):
        observation = env.reset()
        s = None
        tot_reward = 0
        input_trace = torch.zeros((N,))
        E = 0
        
        #input_trace_tab = []
        #output_trace_tab = []
        #input_spike_tab = []
        #output_spike_tab = []
        #trace_z_tab = []
        #xi_tab = []
        #r_tab = []
        #i = 0 
        
        for t in range(100):
            env.render()
            
            input_spike = current_encoding(torch.FloatTensor(observation), param1) #current_encoding
            
#            a = torch.Tensor([[1+torch.sign(torch.FloatTensor(observation)[0]),-1+torch.sign(torch.FloatTensor(observation)[0]),0,0,0,0,0,0],
#                 [0,0,1+torch.sign(torch.FloatTensor(observation)[1]),-1+torch.sign(torch.FloatTensor(observation)[1]),0,0,0,0],
#                 [0,0,0,0,1+torch.sign(torch.FloatTensor(observation)[2]),-1+torch.sign(torch.FloatTensor(observation)[2]),0,0],
#                [0,0,0,0,0,0,1+torch.sign(torch.FloatTensor(observation)[3]),-1+torch.sign(torch.FloatTensor(observation)[3])]])
#            input_spike = current_encoding(torch.matmul(torch.FloatTensor(observation),a), param2) #binary_current_encoding

#            input_spike = place_cell(torch.FloatTensor(observation)) #place_cell
            input_trace = input_trace*O_DECAY + input_spike 
            output_spike,s = forward(input_spike,s,param)
            output_trace = s[3]
            xi = input_trace * output_spike + output_trace * input_spike
            E = E * E_DECAY + xi
            action = int(output_spike.item())
            observation, reward, done, infor = env.step(action)
            tot_reward += reward
            r = get_reward(output_spike,reward)
            dw = ETA * E * r#eward
            param["W"] = param["W"] + dw
            
#            input_spike_tab.append(input_spike[i])
#            output_spike_tab.append(output_spike)
#            input_trace_tab.append(input_trace[i])
#            output_trace_tab.append(output_trace)
#            xi_tab.append(xi[i])
#            trace_z_tab.append(E[i])
#            r_tab.append(r)
        
        env.close()
        tot_reward_tab.append(tot_reward)
#        print(output_spike_tab)    

print(param)
plt.plot(tot_reward_tab)
plt.xlabel('epoch')
plt.ylabel('tot_reward')
plt.show()

#============================================================================        
        #np = 6
        #ax1 = plt.subplot(np, 1, 1)
        #plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.plot(input_spike_tab)
        #plt.ylabel(r'$f_j$')
        #
        #ax2 = plt.subplot(np, 1, 2)
        #plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.plot(output_spike_tab)
        #plt.ylabel(r'$f_i$')
        #
        #ax3 = plt.subplot(np, 1, 3)
        #plt.setp(ax3.get_xticklabels(), visible=False)
        #plt.axhline(linestyle='--', color='b')
        #plt.plot(input_trace_tab)
        #plt.plot(output_trace_tab)
        #plt.ylabel(r'$P_{ij}^+, P_{ij}^-$')
        #
        #ax4 = plt.subplot(np, 1, 4, sharex=ax1)
        #plt.setp(ax4.get_xticklabels(), visible=False)
        #plt.axhline(linestyle='--', color='b')
        #plt.plot(xi_tab)
        #plt.ylabel(r'$\zeta_{ij}$')
        #
        #ax5 = plt.subplot(np, 1, 5, sharex=ax1)
        #plt.setp(ax5.get_xticklabels(), visible=False)
        #plt.axhline(linestyle='--', color='b')
        #plt.plot(trace_z_tab)
        #plt.ylabel(r'$Z_{ij}$')
        #
        #ax6 = plt.subplot(np, 1, 6, sharex=ax1)
        #plt.setp(ax6.get_xticklabels(), visible=False)
        #plt.axhline(linestyle='--', color='b')
        #plt.plot(r_tab)
        #plt.ylabel(r'$R$')
        #plt.show()


