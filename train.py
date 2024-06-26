
from collections import deque
from grid import GridEnvironment
from policy import PolicyNetwork
import torch
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch.nn.functional as F

def select_action(state, policy_net,epsilon=0.1):
    state = state.flatten()
    state_tensor = torch.FloatTensor(state)
    probs = policy_net(state_tensor)

    m = Categorical(probs)
    # print(probs)
    action = m.sample()
    policy_net.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode(eps,optimizer,policy_net):
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
            action = select_action(state,policy_net=policy_net)
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
        finish_episode(episode,optimizer,policy_net)
        if episode % 100 == 98:
            print(f"Episode {episode}, Running Reward: {running_reward:.2f}")
            # rewardslist=[]

def trainagent(gridsize=6):
    
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