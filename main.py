from grid import GridEnvironment
from policy import PolicyNetwork
import torch
import numpy as np
from train import trainagent

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
    trainagent(gridsize)
    input_size=gridsize*gridsize
    output_size=4
    # Load the model using the static method
    loaded_policy_net = PolicyNetwork.load_model('policy_net.pth',input_size,output_size)

    # Print the loaded model to verify
    print(loaded_policy_net)

    # Create a new environment with obstacles
    env_with_obstacles = GridEnvironment(size=gridsize, start=(0, 0), goal=(4, 4), obstacles=[(2,1),(4,1),(2,3),(2,4),(1,5),(4,3),(5,4)])

    # Evaluate the loaded model
    total_reward, path = evaluate_model(env_with_obstacles, loaded_policy_net)

    # Print the results
    print(f"Total Reward: {total_reward}")
    print(f"Path Taken: {path}")
