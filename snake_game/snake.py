from math import sqrt
from typing import List, Any
from enum import Enum
from collections import deque
import numpy as np
import curses
from collections import namedtuple

DEFAULT_SNAKE_BODY = [[0, 0], [0, 1], [0, 2], [0, 3]]


class WorldIcons(Enum):
    EMPTY = 0
    SNAKE = 1
    FOOD = 2


WORLD_ICON_MAP = {
    0: " . ",
    1: " * ",
    2: " # ",
    3: " & ",
}


class Actions(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


class Snake:
    def __init__(self, world_size: int = 20) -> None:
        self._world_size: int = world_size
        self._direction: int = Actions.RIGHT

        # Init basic coords.
        self._body: deque[List[int]] = deque(DEFAULT_SNAKE_BODY)

        # Generate food.
        self.generate_food()

    def generate_food(self) -> None:
        """
        Generate food.

        Returns:
        - List[int]: food position
        """
        while True:
            i, j = np.random.randint(0, self._world_size, 2)
            food = [i, j]

            if food not in self._body:
                self._food_pos = food
                return

    def set_direction(self, dir: int) -> None:
        """
        Set direction of the snake.

        Args:
        - dir: int - direction to set
        """
        self._direction = Actions(dir)

    def _check_limit(self, point: List[int]) -> List[int]:
        """
        Check if point is within the world limits.

        Args:
        - point: List[int] - point to check

        Returns:
        - List[int]: corrected point
        """
        if point[0] > self._world_size - 1:
            point[0] = 0
        elif point[0] < 0:
            point[0] = self._world_size - 1
        elif point[1] < 0:
            point[1] = self._world_size - 1
        elif point[1] > self._world_size - 1:
            point[1] = 0

        return point

    def move(self) -> None:
        """
        Move the snake.
        """
        new_head = self._body[-1].copy()

        if self._direction == Actions.UP:
            new_head[0] -= 1
        elif self._direction == Actions.DOWN:
            new_head[0] += 1
        elif self._direction == Actions.RIGHT:
            new_head[1] += 1
        elif self._direction == Actions.LEFT:
            new_head[1] -= 1

        #new_head = self._check_limit(new_head)

        self._body.append(new_head)
        self._body.popleft()

    def eat_food(self) -> None:
        """
        Eat the food.
        """
        self._body.appendleft(self._body[0])

    def reset(self) -> None:
        """
        Reset the snake.
        """
        self._body = deque(DEFAULT_SNAKE_BODY)
        self.generate_food()

    @property
    def score(self) -> int:
        """
        Get the score of the snake.

        Returns:
        - int: score of the snake
        """
        return len(self._body) - len(DEFAULT_SNAKE_BODY)

    def is_snake_eat_food(self) -> bool:
        """
        Check if the snake is eating the food.

        Returns:
        - bool: True if the snake is eating the food, False otherwise
        """
        return self._body[-1] == self._food_pos

    def render(self, screen: Any) -> None:
        """
        Render the world grid.
        """

        for row in self.get_world_grid():
            screen.addstr("".join([WORLD_ICON_MAP[val] for val in row]) + "\n")

    def get_world_grid(self) -> np.ndarray:
        """
        Get the world grid.

        Returns:
        - np.ndarray: world grid
        """
        world_grid = np.zeros((self._world_size, self._world_size), dtype=int)
        world_grid[self._food_pos[0], self._food_pos[1]] = WorldIcons.FOOD.value
        for coord in self._body:
            world_grid[coord[0], coord[1]] = WorldIcons.SNAKE.value

        return world_grid

    #def get_state(self) -> List[int]:
    #    """
    #    Get the state of the snake.
    #
    #    Returns:
    #    - List[int]: state of the snake
    #    """
    #    state = self.get_world_grid().flatten()
    #    return state.tolist() + [self._direction.value]
    
    def is_collision(self, pt=None):
        if pt is None: #pt is the head of the snake
            pt = self._body[-1]
        if pt[0] > self._world_size - 1 or pt[0] < 0 or pt[1] > self._world_size - 1 or pt[1] < 0:
            return True #if snake hits the side
        if pt in list(self._body)[:-1]:
            return True  #if snake hits itself
        return False
    
    def get_state(self):
        head = self._body[-1]
        
        point_u = head[0] - 1, head[1]
        point_d = head[0] + 1, head[1]
        point_r = head[0], head[1] + 1
        point_l = head[0], head[1] - 1

        dir_l = self._direction == Actions.LEFT
        dir_r = self._direction == Actions.RIGHT
        dir_u = self._direction == Actions.UP
        dir_d = self._direction == Actions.DOWN

        state = [
            (dir_r and self.is_collision(point_r)) or # Danger straight
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            (dir_u and self.is_collision(point_r)) or # Danger right
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            (dir_d and self.is_collision(point_r)) or # Danger left
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            dir_l, #direction
            dir_r,
            dir_u,
            dir_d,

            self._food_pos[0] < self._body[-1][0],  # food left
            self._food_pos[0] > self._body[-1][0],  # food right
            self._food_pos[1] < self._body[-1][1],  # food up
            self._food_pos[1] > self._body[-1][1]  # food down
        ]
        return np.array(state, dtype=int)

    @property
    def distance_to_food(self) -> int:
        head = self._body[-1]
        return abs(head[0] - self._food_pos[0]) + abs(head[1] - self._food_pos[1])
        
def main():
    snake = Snake()
    game_over = False

    # Initialize curses
    stdscr = curses.initscr()
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    while not game_over:
        # Render the game
        stdscr.clear()
        snake.render(stdscr)
        stdscr.refresh()

        # Get user input for direction
        direction = stdscr.getch()
        if direction == ord("w"):
            snake.set_direction(Actions.UP)
        elif direction == ord("s"):
            snake.set_direction(Actions.DOWN)
        elif direction == ord("d"):
            snake.set_direction(Actions.RIGHT)
        elif direction == ord("a"):
            snake.set_direction(Actions.LEFT)

        # Move the snake
        snake.move()

        # Check if the snake is alive
        if snake.is_collision():
            game_over = True
            stdscr.addstr("\nGame Over!\n")
            stdscr.addstr("Score: {}\n".format(snake.score))
            stdscr.refresh()
            snake.reset()

        # Check if the snake is eating the food
        if snake.is_snake_eat_food():
            snake.eat_food()
            snake.generate_food()


if __name__ == "__main__":
    main()
