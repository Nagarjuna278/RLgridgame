import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch.nn.functional as F

class GridEnvironment:
    def __init__(self, size=4, start=(0, 0), goal=(1, 1), obstacles=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.agent_pos = start
        self.grid = np.zeros((self.size, self.size))
        if obstacles is None:
            obstacles=[(1,1),(2,1),(3,1),(4,1),(1,2),(4,2),(2,3),(2,4),(0,5),(5,5)]
        self.obstacles = obstacles
        self.updateobstacles()

    def reset(self, obstaclessize=None):
        self.agent_pos = self.start
        if obstaclessize is None:
            obstaclessize=random.randint(4,max(4,2*self.size))
        self.obstacles=[(random.randint(0,self.size-1),random.randint(0,self.size-1)) for x in range(obstaclessize) ]
        self.updateobstacles()
        while self.is_valid_grid()==False:
            self.obstacles=[(random.randint(0,self.size-1),random.randint(0,self.size-1)) for x in range(obstaclessize) ]
            self.updateobstacles()
        return self.get_state()

    def step(self, action):
        # Define the actions (up, down, left, right)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_pos = (self.agent_pos[0] + actions[action][0], self.agent_pos[1] + actions[action][1])

        # Check if the new position is within bounds and not an obstacle
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and new_pos not in self.obstacles:
            self.agent_pos = new_pos
        else:
            return self.get_state(), -1, False

        # Check if the agent has reached the goal
        done = self.agent_pos == self.goal
        reward = 1 if done else -0.1  # Encourage progress and discourage staying still

        return self.get_state(), reward, done

    def updateobstacles(self, obstacles=None):
        self.grid = np.zeros((self.size, self.size))  # Reset the grid
        if obstacles is not None:
            self.obstacles = obstacles
        if self.goal in self.obstacles:
            self.obstacles.remove(self.goal)
        if self.start in self.obstacles:
            self.obstacles.remove(self.start)
        
        for obs in self.obstacles:
            self.grid[obs] = -1  # Mark obstacles

    def is_valid_grid(self):
        grid = self.grid.copy()
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        queue = deque([(start_row, start_col)])
        visited = set((start_row, start_col))
        while queue:
            row, col = queue.popleft()
            if row == goal_row and col == goal_col:
                return True
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
            for direction in directions:
                new_row, new_col = row + direction[0], col + direction[1]
                if (new_row >= 0 and new_row < len(grid) and new_col >= 0 and new_col < len(grid[0]) and
                    (new_row, new_col) not in visited and grid[new_row][new_col] != -1):
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
        return False # Placeholder, implement actual pathfinding check

    def get_state(self):
        state = self.grid.copy()
        state[self.agent_pos] = 1  # Mark agent position
        state[self.goal] = 2  # Mark goal position
        return state

    def render(self):
        env = self.get_state().copy()
        plt.imshow(env, cmap='coolwarm')
        plt.show(block=False)
        plt.pause(0.2)
        plt.clf()
        plt.close()

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,output_size)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x

    @staticmethod
    def load_model(filepath, input_size, output_size):
        model = PolicyNetwork(input_size, output_size)
        model.load_state_dict(torch.load(filepath))
        return model


def select_action(state,epsilon=0.1):
    state = state.flatten()
    state_tensor = torch.FloatTensor(state)
    probs = policy_net(state_tensor)

    m = Categorical(probs)
    # print(probs)
    action = m.sample()
    policy_net.saved_log_probs.append(m.log_prob(action))
    return action.item()

from collections import deque

def finish_episode(eps,optimizer):
    gamma = 0.90
    R = 0
    policy_loss = []
    returns = deque()
    
    # Calculate the returns (discounted rewards)
    for r in policy_net.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    
    # Normalize the returns
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    # Calculate policy loss
    for log_prob, R in zip(policy_net.saved_log_probs, returns):
        policy_loss.append(-log_prob * R.unsqueeze(0))
    
    # Perform a single backpropagation step
    optimizer.zero_grad()
    if policy_loss:
        policy_loss = torch.cat(policy_loss).sum()
        # print(policy_loss)
        policy_loss.backward()
    optimizer.step()
    
    # Clear the rewards and log probabilities
    del policy_net.rewards[:]
    del policy_net.saved_log_probs[:]
    return 0





def train(env, policy_net, optimizer, num_episodes=100000):
    running_reward = 100
    max_steps_per_episode = env.size*env.size*env.size
    # rewardslist=[]

    for episode in range(num_episodes):
        # print(f"Episode {episode}")
        if episode%100 == 99:
            obstaclessize=random.randint(4,max(4,2*env.size))
            state = env.reset(obstaclessize=obstaclessize)
            env.updateobstacles()
            env.render()
        else:
            env.agent_pos=(0,0)
            state=env.get_state()
        

        if episode%100 == 0:
            print(episode)
        ep_reward = 0
        policy_net.rewards = []
        policy_net.saved_log_probs = []
        visited_states = set()
        
        # Run one episode
        done = False
        for t in range(1, max_steps_per_episode):
            action = select_action(state)
            next_state, reward, done = env.step(action)
            
            # Penalize revisiting states
            if tuple(next_state.flatten()) in visited_states:
                reward -= 1
            else:
                visited_states.add(tuple(next_state.flatten()))
            
            if episode%100 == 98:
                env.render()
            policy_net.rewards.append(reward)
            ep_reward += reward
            state = next_state
            
            if done:
                break
            
        running_reward = ep_reward
        finish_episode(episode,optimizer)
        if episode % 100 == 98:
            print(f"Episode {episode}, Running Reward: {running_reward:.2f}")
            # rewardslist=[]

def evaluate_model(env, model):
    state = env.get_state()
    total_reward = 0
    done = False
    path = [env.start]

    while not done:
        env.render()
        state = np.array(state).flatten()
        state_tensor = torch.FloatTensor(state)
        probs = model(state_tensor)
        action = torch.argmax(probs).item()
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(env.agent_pos)
        state = next_state

    return total_reward, path

if __name__ == "__main__":
    # Define the grid size and action space size
    gridsize=6
    input_size = gridsize*gridsize  # Grid size (4x4)
    output_size = 4     # Number of actions (up, down, left, right)

    # Initialize the policy network
    policy_net = PolicyNetwork(input_size, output_size)

    # Define the optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

    # Training the model (assuming the train function is defined)
    env = GridEnvironment(size=gridsize, start=(0, 0), goal=(4, 4),obstacles=[(2,1),(4,1),(2,3),(2,4),(1,5),(4,3),(5,4)])
    train(env, policy_net, optimizer, num_episodes=1000000)

    # Save the trained model
    torch.save(policy_net.state_dict(), 'policy_net.pth')

    # Load the model using the static method
    loaded_policy_net = PolicyNetwork.load_model('policy_net.pth', input_size, output_size)

    # Print the loaded model to verify
    print(loaded_policy_net)

    # Create a new environment with obstacles
    env_with_obstacles = GridEnvironment(size=gridsize, start=(0, 0), goal=(4, 4), obstacles=[(2,1),(4,1),(2,3),(2,4),(1,5),(4,3),(5,4)])

    # Evaluate the loaded model
    total_reward, path = evaluate_model(env_with_obstacles, loaded_policy_net)

    # Print the results
    print(f"Total Reward: {total_reward}")
    print(f"Path Taken: {path}")
