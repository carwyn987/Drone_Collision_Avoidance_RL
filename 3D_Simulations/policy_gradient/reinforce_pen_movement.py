import random
import airsim
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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
        if img_rgb.shape == (84, 84, 3):
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
    def __init__(self, env, model, roll_model):
        self.env = env
        self.current_ep_return = 0
        self.model = model
        self.roll_model = roll_model

    # Takes and executes an action, returns tuple (next state, reward, done)
    def executeAction(self, throttle, roll, action_duration):
        t = throttle.item()
        r = roll.item()

        # Send action to UE4
        # roll_rate, pitch_rate, yaw_rate, throttle, duration
        moveRequest = self.env.client.moveByAngleRatesThrottleAsync(r, 0, 0, t, action_duration).join()

        (done, reward, next_state) = self.env.step()

        self.current_ep_return += reward
        return (done, reward)

    # Restarts environment, and resets all necessary variables
    def restart_episode(self):
        self.env.reset()
        self.current_ep_return = 0

    # Define epsilon greedy policy
    def choose_stoch_action(self, device="cpu"):
        if self.env.bufferFull():
            state_numpy_array = np.array([self.env.observation_buffer], copy=False, dtype="float32")
            state_tensor = torch.tensor(state_numpy_array).to(device)
            mean, variance = self.model(state_tensor)
            m = torch.distributions.Normal(mean, torch.sqrt(variance))
            action = m.sample()
            log_prob = m.log_prob(action)

            mean_roll, var_roll = self.roll_model(state_tensor)
            m_roll = torch.distributions.Normal(mean_roll, torch.sqrt(var_roll))
            action_roll = m_roll.sample()
            log_prob_roll = m_roll.log_prob(action_roll)

            flag = False

            return flag, action.cpu().numpy(), log_prob, action_roll.cpu().numpy(), log_prob_roll
        else:
            zero_t = torch.tensor([[0.0]]).to(device)
            half_t = torch.tensor([[0.5]]).to(device)
            one_t = torch.tensor([[1.0]]).to(device)
            three_t = torch.tensor([[3.0]]).to(device)
            m = torch.distributions.Normal(half_t, one_t)
            action = m.sample()
            log_prob = m.log_prob(action)

            m_roll = torch.distributions.Normal(zero_t, three_t)
            action_roll = m_roll.sample()
            log_prob_roll = m_roll.log_prob(action_roll)

            flag = True

            return flag, action.cpu().numpy(), log_prob, action_roll.cpu().numpy(), log_prob_roll

# Normalize image values
def normalizeValues(obs):
    return np.array(obs).astype(np.float32) / 255.0

class CNN_Action_Model(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()

        if input_dim[1] != 84:
            raise ValueError(f"Expecting input height: 84, got: {input_dim[1]}")
        if input_dim[2] != 84:
            raise ValueError(f"Expecting input width: 84, got: {input_dim[2]}")

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size)
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        trunk_out = self.trunk(input)

        mu = self.mean(trunk_out)
        variance = self.var(trunk_out)

        return mu, variance+0.01

start = time.time()

## Set up and hyperparameter selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer_length = 4
input_size = (buffer_length, 84, 84)
num_training_episodes = 15000
num_actions = 1 # 0, 0.1, 0.2, ... 0.9, 1
hidden_size = 256

# Epsilon parameters
gamma = 0.9
lr = 0.0001
action_duration = 0.1 # seconds

model = CNN_Action_Model(input_size, hidden_size, num_actions).to(device)
model_roll = CNN_Action_Model(input_size, hidden_size, num_actions).to(device)

env = Environment(buffer_length)
agent = Agent(env, model, model_roll)

# Define optimizer as Adam, with our learning rate
optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer_roll = optim.Adam(model.parameters(), lr=lr)

# Instantiate logging lists
return_save = []
ep_return = []
ep_loss = []
ep_loss_roll = []

for ep in range(num_training_episodes):

    # Define lists for saving trajectory
    probs = []
    prob_rolls = []
    rewards = []
    actions = []

    done = False
    step = 0
    # Run a single episode
    while not done:
        # Choose an action
        isRand, action, prob, action_roll, prob_roll = agent.choose_stoch_action(device)

        # Execute action in environment and get a next state
        done, reward = agent.executeAction(action, action_roll, action_duration)

        step += 1

        if not isRand:
            # Append data to lists
            actions.append(action)
            probs.append(prob)
            prob_rolls.append(prob_roll)
            rewards.append(reward)

    return_save.append(agent.current_ep_return)
    print(f"Episode {ep}, Step {step}, Total Reward: {agent.current_ep_return}")
    agent.restart_episode()

    if len(probs) != 0:

        total_reward = 0
        return_data = np.zeros(len(rewards), dtype="float32")
        for i in list(reversed(range(len(rewards)))):
            total_reward = total_reward * gamma + rewards[i]
            return_data[i] = total_reward
        return_tensor = torch.tensor(return_data, device=device)

        log_prob = torch.stack(probs)
        log_prob_roll = torch.stack(prob_rolls)
        score = torch.sum(-log_prob * return_tensor)
        score_roll = torch.sum(-log_prob_roll * return_tensor)


        # print(score, log_prob, return_tensor)
        optimizer.zero_grad()
        score.backward()
        optimizer.step()
        ep_loss.append(score.cpu().detach().numpy().item())

        optimizer_roll.zero_grad()
        score_roll.backward()
        optimizer_roll.step()
        ep_loss_roll.append(score.cpu().detach().numpy().item())

        agent.restart_episode()
        tot_reward = sum(rewards)
        ep_return.append(tot_reward)

    if np.mean(ep_return[-20:]) > 400 and np.mean(ep_return[-5:]) >= 800:
        print("Converged to maximum possible reward (1000)")
        break

    if ep % 50 == 0 and ep > 0:
        print(f'Episode {ep}, Average reward for 50 recent runs: {np.mean(ep_return[-50:])}')

end = time.time()
print("Time to learn : ", end - start)


# Save model
torch.save(model.state_dict(), "../models/reinforce_dodge_full_gradient_throttle.pth")
torch.save(model_roll.state_dict(), "../models/reinforce_dodge_full_gradient_roll.pth")