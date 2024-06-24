import random
from typing_extensions import Self

class Player:
    def __init__(self, grid, x, y, player_type):
        self.grid = grid
        self.x = x
        self.y = y
        self.type = player_type

    def move(self):
        raise NotImplementedError

    def attempt_move(self, new_x, new_y):
        if self.grid.is_within_bounds(new_x, new_y) and not self.grid.is_blocked(new_x, new_y) and not self.grid.is_occupied(new_x, new_y):
            self.grid.vacate(self.x, self.y)
            self.x, self.y = new_x, new_y
            self.grid.occupy(self.x, self.y, self)
            print("returning true")
            return True
        print("returning false",self.type)
        return False

class GoalPlayer(Player):
    def __init__(self, grid, x, y, goal_x, goal_y):
        super().__init__(grid, x, y, GoalPlayer)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = []

    def move(self):
        if not self.path:
            self.path = self.grid.a_star((self.x, self.y), (self.goal_x, self.goal_y))

        if self.path:
            next_move = self.path.pop(0)
            if self.attempt_move(next_move[0], next_move[1]):
                if self.x == self.goal_x and self.y == self.goal_y:
                    gameheight,gamewidth = self.grid.dimensions()
                    self.goal_x = 0 if self.goal_x ==  gamewidth-1 else gamewidth -1
                    self.goal_y = 0 if self.goal_y == gameheight-1 else gameheight-1
            else:
                print("in else")
                # If the next move is blocked or occupied, try moving in one of the 4 directions
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                random.shuffle(directions)
                print(self.x,self.y)
                for dx, dy in directions:
                    print("trying ",dx,dy)
                    new_x, new_y = self.x + dx, self.y + dy
                    if self.attempt_move(new_x,new_y):
                        break
                print(self.x,self.y)

class RandomPlayer(Player):
    def __init__(self, grid, x, y):
        super().__init__(grid, x, y, RandomPlayer)

    def move(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)
        for dx, dy in directions:
            if self.attempt_move(self.x + dx, self.y + dy):
                break

class DisruptPlayer(Player):
    def __init__(self, grid, x, y, target_players):
        super().__init__(grid, x, y, DisruptPlayer)
        self.target_players = [p for p in target_players if isinstance(p, (RandomPlayer, GoalPlayer))]

    def move(self):
        print(Player)
        self.target_players = sorted(
            [p for p in self.target_players if isinstance(p, (RandomPlayer, GoalPlayer))],
            key=lambda p: isinstance(p, GoalPlayer),
            reverse=True
        )
        if self.target_players:
            target = self.target_players[0]
            dx = max(min(target.x - self.x, 1), -1)  # Determine direction along x-axis
            dy = max(min(target.y - self.y, 1), -1)  # Determine direction along y-axis
            if self.attempt_move(self.x + dx, self.y + dy):
              return
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)
        for dx1, dy1 in directions:
            if self.attempt_move(self.x + dx1, self.y + dy1):
                print("disruptplayer")
                break
