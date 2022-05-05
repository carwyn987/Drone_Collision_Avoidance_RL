import random
import airsim
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque

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
        ep_done = self.client.simGetCollisionInfo().has_collided

        return ep_done, next_state


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

        t = throttle / 10.0
        r = (roll - 5.0) / 2.0

        # Send action to UE4
        # roll_rate, pitch_rate, yaw_rate, throttle, duration
        moveRequest = self.env.client.moveByAngleRatesThrottleAsync(r, 0, 0, t, action_duration).join()

        done, _ = self.env.step()

        return done

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
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        return self.net(input)

## Set up and hyperparameter selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

num_episodes = 5
max_steps = 180
action_duration = 0.1 # seconds
buffer_length = 4
gamma = 0.99
input_size = (buffer_length, 84, 84)
num_actions = 11 # 0, 0.1, 0.2, ... 0.9, 1

# Epsilon parameters
epsilon = 0

# Initialize network
model_dir = "models/"
model_path = "dqn_dodge_pen_movement_throttle.pth"
model = CNN_Action_Model(input_size, num_actions).to(device)
model.load_state_dict(torch.load(model_dir + model_path))
model.eval()

model_path2 = "dqn_dodge_pen_movement_roll.pth"
model2 = CNN_Action_Model(input_size, num_actions).to(device)
model2.load_state_dict(torch.load(model_dir + model_path2))
model2.eval()

env = Environment(buffer_length)
agent = Agent(env, model, model2, None)

# Let's log the returns, and step in episode
return_save = []
episode_num = 0
step_num = 0

for episode_num in range(num_episodes):
    while True:
        # Choose an action
        action = agent.choose_stoch_action(epsilon, device)
        # Execute action in environment and get a next state
        done = agent.executeAction(action, action_duration)

        step_num += 1

        if done or max_steps == step_num:
            return_save.append(step_num)
            episode_num += 1
            print(f"Episode {episode_num}, Step reached: {step_num}")
            agent.restart_episode()
            step_num = 0
            break

print(f"Model performance on {num_episodes} test episodes averaged reward = {np.mean(return_save)} with maximum possible reward of {max_steps}")