import curses
from collections import deque
from enum import Enum
from typing import Any, List

import numpy as np

DEFAULT_SNAKE_BODY = [[0, 0], [0, 1], [0, 2], [0, 3]]


WORLD_ICON_MAP = {
    0: " . ",
    1: " * ",
    2: " # ",
}


class WorldIcons(Enum):
    EMPTY = 0
    SNAKE = 1
    FOOD = 2


class Actions(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


class Snake:
    """
    Snake class to represent the snake game.
    """

    def __init__(self, world_size: int = 10) -> None:
        """
        Initialize the snake game.

        Args:
        - world_size: int - size of the world grid
        """
        self._world_size: int = world_size
        self._direction: int = Actions.RIGHT

        # Init basic coords.
        self._body: deque[List[int]] = deque(DEFAULT_SNAKE_BODY)

        # Generate food.
        self.generate_food()

    def generate_food(self) -> None:
        """
        Generate food for the snake.
        """
        while True:
            i, j = np.random.randint(0, self._world_size, 2)
            food = [i, j]

            if food not in self._body:
                self._food_pos = food
                return

    def set_direction(self, dir: int | Actions) -> None:
        """
        Set the direction of the snake.

        Args:
        - dir: int | Actions - direction of the snake
        """
        self._direction = dir if isinstance(dir, Actions) else Actions(dir)

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
        food_x, food_y = self._food_pos
        world_grid[food_x, food_y] = WorldIcons.FOOD.value
        for coord in self._body:
            world_grid[coord[0], coord[1]] = WorldIcons.SNAKE.value

        return world_grid

    def is_collision(self, head=None):
        """
        Check if the snake is colliding with the wall or itself.

        Args:
        - head: List[int] - head of the snake

        Returns:
        - bool: True if the snake is colliding, False otherwise
        """
        if head is None:
            head = self._body[-1]
        if (
            head[0] > self._world_size - 1
            or head[0] < 0
            or head[1] > self._world_size - 1
            or head[1] < 0
        ):
            return True
        if head in list(self._body)[:-1]:
            return True
        return False

    def get_state(self) -> np.ndarray:
        """
        Get the state of the snake.

        Returns:
        - np.ndarray: state of the snake

        The state of the snake is represented as a 11-dimensional vector:
        - danger_straight: True if there is danger straight ahead, False otherwise
        - danger_left: True if there is danger to the left, False otherwise
        - danger_right: True if there is danger to the right, False otherwise
        - direction_up: True if the snake is moving up, False otherwise
        - direction_down: True if the snake is moving down, False otherwise
        - direction_left: True if the snake is moving left, False otherwise
        - direction_right: True if the snake is moving right, False otherwise
        - food_up: True if the food is above the snake, False otherwise
        - food_down: True if the food is below the snake, False otherwise
        - food_left: True if the food is to the left of the snake, False otherwise
        - food_right: True if the food is to the right of the snake, False otherwise
        """

        # Get the head of the snake.
        head = self._body[-1]

        # Points in the direction the snake is heading.
        point_up = head[0] - 1, head[1]
        point_down = head[0] + 1, head[1]
        point_left = head[0], head[1] - 1
        point_right = head[0], head[1] + 1

        # Directions.
        direction_up = self._direction == Actions.UP
        direction_down = self._direction == Actions.DOWN
        direction_left = self._direction == Actions.LEFT
        direction_right = self._direction == Actions.RIGHT

        # Check for danger in each direction.
        danger_straight = (
            (direction_right and self.is_collision(point_right))
            or (direction_left and self.is_collision(point_left))
            or (direction_up and self.is_collision(point_up))
            or (direction_down and self.is_collision(point_down))
        )
        danger_left = (
            (direction_down and self.is_collision(point_right))
            or (direction_up and self.is_collision(point_left))
            or (direction_right and self.is_collision(point_up))
            or (direction_left and self.is_collision(point_down))
        )
        danger_right = (
            (direction_up and self.is_collision(point_right))
            or (direction_down and self.is_collision(point_left))
            or (direction_left and self.is_collision(point_up))
            or (direction_right and self.is_collision(point_down))
        )

        # Food location.
        food_up = self._food_pos[0] < head[0]
        food_down = self._food_pos[0] > head[0]
        food_left = self._food_pos[1] < head[1]
        food_right = self._food_pos[1] > head[1]

        # State.
        state = [
            danger_straight,
            danger_left,
            danger_right,
            direction_up,
            direction_down,
            direction_left,
            direction_right,
            food_up,
            food_down,
            food_left,
            food_right,
        ]

        return np.array(state, dtype=int)

    @property
    def distance_to_food(self) -> int:
        """
        Get the distance to the food.

        Returns:
        - int: distance to the food

        The distance to the food is calculated as the Manhattan distance between the head of the snake and the food.
        """
        head = self._body[-1]
        return abs(head[0] - self._food_pos[0]) + abs(head[1] - self._food_pos[1])


def main():
    snake = Snake()
    game_over = False

    # Initialize curses.
    stdscr = curses.initscr()
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(200)

    while not game_over:
        # Render the game.
        stdscr.clear()
        snake.render(stdscr)
        stdscr.refresh()

        # Get user input for direction.
        direction = stdscr.getch()
        if direction == ord("w"):
            snake.set_direction(Actions.UP)
        elif direction == ord("s"):
            snake.set_direction(Actions.DOWN)
        elif direction == ord("d"):
            snake.set_direction(Actions.RIGHT)
        elif direction == ord("a"):
            snake.set_direction(Actions.LEFT)

        # Move the snake.
        snake.move()

        # Check if the snake is alive.
        if snake.is_collision():
            game_over = True
            stdscr.addstr("\nGame Over!\n")
            stdscr.addstr("Score: {}\n".format(snake.score))
            stdscr.refresh()

        # Check if the snake is eating the food.
        if snake.is_snake_eat_food():
            snake.eat_food()
            snake.generate_food()


if __name__ == "__main__":
    main()
