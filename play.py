from main import PolicyNetwork, GridEnvironment
import numpy as np
import torch


def evaluate_model(env, model):
    state = env.reset()
    total_reward = 0
    done = False
    env.obstacles=[(0,1),(1,1),(3,1),(3,2)]
    state=env.get_state()
    path = [env.start]
    state = np.array(state).flatten()
    state_tensor = torch.FloatTensor(state)
    probs = model(state_tensor)
    print(probs)
    while not done:
        state = np.array(state).flatten()
        state_tensor = torch.FloatTensor(state)
        probs = model(state_tensor)
        action = torch.argmax(probs).item()
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(env.agent_pos)
        state = next_state

    return total_reward, path



model_state_dict= torch.load('policy_net.pth')


loaded_env = GridEnvironment(size=4, start=(0, 0), goal=(3, 3), obstacles=[(0, 1), (1, 1), (3, 0), (3, 2)])
loaded_policy_net = PolicyNetwork(16,4)
loaded_policy_net.load_state_dict(model_state_dict)
# Evaluate the loaded model
total_reward, path = evaluate_model(loaded_env, loaded_policy_net)

# Print the results
print(f"Total Reward: {total_reward}")
print(f"Path Taken: {path}")