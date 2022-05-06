#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import random

import airsim
import numpy as np
import os
import tempfile
import pprint
import cv2
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from airsim import string_to_uint8_array
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[2]:

# Define environment to encapsulate connection with UE4
class Environment():
    def __init__(self, buffer_length):
        self.connectUE4()
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.observation_buffer = deque(maxlen=buffer_length)
        self.max_buffer_length = buffer_length
        self.reset()
    
    def connectUE4(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
    def disconnectUE4(self):
        self.client.reset()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        im = self.getImage()
        while im is None:
            im = self.getImage()
        self.observation_buffer = deque(maxlen=buffer_length)
        return im

    def bufferFull(self):
        return len(self.observation_buffer) == self.max_buffer_length
    
    def getState(self, device):
        return torch.tensor(np.expand_dims(np.array(list(self.observation_buffer), dtype="float32"), axis=0)).to(device)
    
    def getImage(self):
        # Request image
        # responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis)])
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, False, False)])
        response = responses[0]

        # get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # original image is flipped vertically
        img_rgb = np.flipud(img_rgb)
        # turn image to grayscale, normalize
        if img_rgb.shape == (84,84,3):
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            img_normal_gray = normalizeValues(img_gray)

            self.observation_buffer.append(img_normal_gray)
        else:
            img_normal_gray = None
                        
        return img_normal_gray
    
    def step(self):
        # Get observation image
        next_state = self.getImage()
        while next_state is None:
            next_state = self.getImage()

        self.observation_buffer.append(next_state)
                
        # Calculate reward
        drone_state = self.client.getMultirotorState()
        drone_position_vector = drone_state.kinematics_estimated.position
        drone_vel_vector = drone_state.kinematics_estimated.linear_velocity
        ep_done = self.client.simGetCollisionInfo().has_collided

        (r, ep_done) = computeReward(drone_position_vector, ep_done, drone_vel_vector)

        return ep_done, r, next_state


# In[3]:

# Define an agent to encapsulate interaction with the environment
class Agent:
    def __init__(self, env, model, roll_model, replay_buffer):
        self.env = env
        self.current_ep_return = 0
        self.model = model
        self.roll_model = roll_model
        self.replay_buffer = replay_buffer
        
    # Takes and executes an action, returns tuple (next state, reward, done)
    def executeAction(self, a, action_duration):
        throttle, roll = a

        t = throttle/10.0
        r = (roll - 5.0)/2.0
        
        # Send action to UE4
        # roll_rate, pitch_rate, yaw_rate, throttle, duration
        moveRequest = self.env.client.moveByAngleRatesThrottleAsync(r, 0, 0, t, action_duration).join()

        bufFull = False
        if self.env.bufferFull():
            oldStateSave = np.array(self.env.observation_buffer)
            bufFull = True

        (done, reward, next_state) = self.env.step()

        if bufFull:
            sample_tuple = SingleExperience(oldStateSave, throttle, roll, reward, done, np.array(self.env.observation_buffer))
            self.replay_buffer.append(sample_tuple)
        
        self.current_ep_return += reward
        return (done, reward)

    # Restarts environment, and resets all necessary variables
    def restart_episode(self):
        self.env.reset()
        self.current_ep_return = 0

    # Define epsilon greedy policy
    def choose_stoch_action(self, epsilon, device="cpu"):
        if self.env.bufferFull() and random.uniform(0, 1) > epsilon:
            state_numpy_array = np.array([self.env.observation_buffer], copy=False)
            state_tensor = torch.tensor(state_numpy_array).to(device)
            model_estimated_action_values = self.model(state_tensor)
            model_est_roll_vals = self.roll_model(state_tensor)
            act_v = torch.argmax(model_estimated_action_values, dim=1)  # This is the same as torch.argmax
            act_roll = torch.argmax(model_est_roll_vals, dim=1)
            action = int(act_v.item()), int(act_roll.item())
        else:
            action = np.random.choice(np.arange(0, 11, 1)), np.random.choice(np.arange(0, 11, 1))
        return action

# In[4]:


# Define helper functions

# Normalize image values
def normalizeValues(obs):
    return np.array(obs).astype(np.float32) / 255.0

# Determine a reward given the drone position and position
def computeReward(pos_vector, collision, drone_vel_vector):
    if not collision and abs(pos_vector.z_val) < 5 and abs(pos_vector.z_val) > 1:
        negateMovement = sigmoid(drone_vel_vector.y_val + drone_vel_vector.z_val)
        return (1.0 - negateMovement) + 0.1, False
    elif not collision and abs(pos_vector.z_val) < 1:
        return 0.01, False
    elif abs(pos_vector.z_val) > 15:
        return 0, True
    else:
        return 0, True

# In[5]:


# Let us now define our replay buffer
# Remember that the replay buffer is simply a datastore in which we can keep samples
# taken from interacting with our environment, and store them such that later on
# we can randomly generate a batch to train our neural network value approximation.

# This will be implemented using a deque data structure

# First let us define one individual sample as a namedtuple
# A NamedTuple is simply a factory function for creating tuple subclasses with
# naming on each field.
SingleExperience = namedtuple('SingleExperience', 
                              field_names = ['state','throttle','roll','reward','done','nextstate'])

# Now let us define the Replay Buffer
# More info about deque here : https://pythontic.com/containers/deque/introduction
class ReplayBuffer:
    def __init__(self,size):
        self.buffer = deque(maxlen = size)
    def sampleBuf(self,size):
        # First let us get a list of elements to sample
        # Make sure to choose replace = False so that we cannot choose the same
        # sample tuples multiple times
        el_inds = np.random.choice(len(self.buffer), size, replace=False)

        # A nifty piece of code implemented by @Jordi Torres, this makes use of the
        # zip function to combine each respective data field, and returns the np arrays
        # of each separate entity.
        arr_chosen_samples = [self.buffer[i] for i in el_inds]
        # Take the samples and break them into their respective iterables
        state_arr, throttle_arr, roll_arr, reward_arr, done_arr, next_state_arr = zip(*arr_chosen_samples)
        # Return these iteratables as np arrays of the correct types
        return np.array(state_arr),np.array(throttle_arr),np.array(roll_arr),np.array(reward_arr,dtype=np.float32),np.array(done_arr, dtype=np.uint8),np.array(next_state_arr)

    def append(self, sample):
        self.buffer.append(sample)

    def size(self):
        return len(self.buffer)

class CNN_Action_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim[1] != 84:
            raise ValueError(f"Expecting input height: 84, got: {input_dim[1]}")
        if input_dim[2] != 84:
            raise ValueError(f"Expecting input width: 84, got: {input_dim[2]}")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):
        return self.net(input)

## Set up and hyperparameter selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

num_training_episodes = 1000
action_duration = 0.1 # seconds
buffer_length = 4
max_steps = 200000
gamma = 0.99
lr = 0.0001
input_size = (buffer_length, 84, 84)
# output_size = 2
variance = torch.tensor(0.5).to(device)
num_actions = 11 # 0, 0.1, 0.2, ... 0.9, 1

# Set up replay buffer parameters
buffer_size = 20000
batch_size = 64

# Epsilon parameters
epsilon = 1
decay = 0.99998
min_epsilon = 0.02

# Define learning rate and other learning parameters
sync_target_frequency = 1000

# Initialize networks
behavior_model = CNN_Action_Model(input_size, num_actions).to(device)
target_model = CNN_Action_Model(input_size, num_actions).to(device)

behavior_model_roll = CNN_Action_Model(input_size, num_actions).to(device)
target_model_roll = CNN_Action_Model(input_size, num_actions).to(device)

# Initialize network as learned network
# model_dir = "models/"
# model_path1 = "dqn_dodge_behavior_throttle_full_yz.pth"
# model_path2 = "dqn_dodge_behavior_roll_full_yz.pth"
# behavior_model.load_state_dict(torch.load(model_dir + model_path1))
# behavior_model_roll.load_state_dict(torch.load(model_dir + model_path2))

# Initialize the buffer and agent
replay_buffer = ReplayBuffer(buffer_size)

env = Environment(buffer_length)
agent = Agent(env, behavior_model, behavior_model_roll, replay_buffer)

# Define optimizer as Adam, with our learning rate
optimizer = optim.Adam(behavior_model.parameters(), lr=lr)
optimizer_roll = optim.Adam(behavior_model_roll.parameters(), lr=lr)


# Let's log the returns, and step in episode
return_save = []
loss_save = []
episode_num = 0

for step in range(max_steps):
    # Choose an action
    action = agent.choose_stoch_action(epsilon, device)
    # Execute action in environment and get a next state
    done, reward = agent.executeAction(action, action_duration)
    if done:
        return_save.append(agent.current_ep_return)
        episode_num += 1
        print(f"Episode {episode_num}, Step {step}, Total Reward: {agent.current_ep_return}")
        agent.restart_episode()

    # # Implement early stopping
    # if np.average(return_save[-5:]) >= 19.5:
    #    break
    
    # Decay epsilon
    epsilon *= decay
    if epsilon < min_epsilon:
        epsilon = min_epsilon

    # Sync target network
    if step % sync_target_frequency == 0:
        # Copy weights to target network
        target_model.load_state_dict(behavior_model.state_dict())
        target_model_roll.load_state_dict(behavior_model_roll.state_dict())

    # Train DQN

    # if we have enough data in buffer
    if replay_buffer.size() > 2*batch_size:

        # First, sample from the replay buffer
        cur_state_arr, throttle_arr, roll_arr, reward_arr, done_arr, next_state_arr = replay_buffer.sampleBuf(batch_size)
        # cur_state_arr = np.expand_dims(cur_state_arr, axis=1)
        # print(cur_state_arr.shape, action_arr.shape, reward_arr.shape, done_arr.shape, next_state_arr.shape) # RIGHT NOW SHAPE IS NOT 4 samples each training sample...

        # Copy the arrays to the GPU as tensors.
        # This allows the GPU to be used for computation speed improvements.
        # Follow guide https://discuss.pytorch.org/t/converting-numpy-array-to-tensor-on-gpu/19423/3 
        # and
        # https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c 
        # and originally
        # https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/tree/master/Chapter04 
        cur_state_tensor = torch.tensor(cur_state_arr).to(device)
        action_tensor = torch.tensor(throttle_arr).to(device)
        roll_tensor = torch.tensor(roll_arr).to(device)
        reward_tensor = torch.tensor(reward_arr).to(device)
        done_tensor_mask = torch.ByteTensor(done_arr).to(device)
        next_state_tensor = torch.tensor(next_state_arr).to(device)

        # Now that we have the tensorized versions of our separated batch data, we
        # must pass the cur_states into the behavior model to get the values of the 
        # taken actions.
        
        # First pass the batch into the model
        beh_model_output_cur_state = behavior_model(cur_state_tensor)
        roll_model_output = behavior_model_roll(cur_state_tensor)

        # Now we must process this tensor and extract the Q-values for taken actions.
        # This is done with a pretty magical command that was constructed by 
        # Maxim Lapan, source in the resources section of hw document
        estimated_taken_action_vals = beh_model_output_cur_state.gather(1, action_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        estimated_taken_action_vals_roll = roll_model_output.gather(1, roll_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        # Note that this should return a 1d tensor of action values taken

        # Now we must calculate the target value

        # First we must calculate the predicted value of taking the max action at the
        # next state. This is because we are following the equation of format:
        # Value of (state,action) = reward + discount * max(Q(s',a'))
        # with the last term calculated from the target network
        max_next_action_value_tensor = target_model(next_state_tensor).max(1)[0]
        max_next_action_value_tensor_roll = target_model_roll(next_state_tensor).max(1)[0]
        # Note that max(1)[0] gets the maximum value in each batch sample, hence
        # getting the max from dimension 1. [0] just extracts the value.

        # Now we must mask the done values such that reward is 0.
        max_next_action_value_tensor[done_tensor_mask] = float(0)
        max_next_action_value_tensor_roll[done_tensor_mask] = float(0)

        target_values = max_next_action_value_tensor.detach() * gamma + reward_tensor
        target_values_roll = max_next_action_value_tensor_roll.detach() * gamma + reward_tensor

        # Calculate loss
        loss = nn.MSELoss()(estimated_taken_action_vals, target_values)
        loss_roll = nn.MSELoss()(estimated_taken_action_vals_roll, target_values_roll)

        if step % 100 == 0:
            loss_save.append(loss.item())

        # Perform back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_roll.zero_grad()
        loss_roll.backward()
        optimizer_roll.step()


# In[ ]:

# Save model
torch.save(behavior_model.state_dict(), "models/dqn_dodge_pen_movement_throttle.pth")
torch.save(behavior_model_roll.state_dict(), "models/dqn_dodge_pen_movement_roll.pth")