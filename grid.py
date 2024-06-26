import matplotlib.pyplot as plt
import random
from collections import deque
import numpy as np

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