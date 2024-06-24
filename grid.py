import heapq

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.blocks = set()

    def dimensions(self):
        return self.height,self.width

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        return (x, y) in self.blocks

    def add_block(self, x, y):
        if self.is_within_bounds(x, y):
            self.blocks.add((x, y))

    def is_occupied(self, x, y):
        return self.grid[y][x] is not None

    def occupy(self, x, y, player):
        self.grid[y][x] = player

    def vacate(self, x, y):
        self.grid[y][x] = None

    def neighbors(self, x, y):
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        neighbors = filter(lambda pos: self.is_within_bounds(pos[0], pos[1]) and not self.is_blocked(pos[0], pos[1]), neighbors)
        return neighbors

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.neighbors(*current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []
