from enum import Enum

import numpy as np
import pygame

GAME_SPEED = 600

class Actions(Enum):
    LEFT_UP = 0
    LEFT_DOWN = 1
    RIGHT_UP = 2
    RIGHT_DOWN = 3


class Pong:
    def __init__(self):
        self.width: int = 800
        self.height: int = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running: bool = True
        self.paddle_width: int = 10
        self.paddle_height: int = 100
        self.paddle_speed: int = 10
        self.paddle_left_y: int = self.height // 2 - self.paddle_height // 2
        self.paddle_right_y: int = self.height // 2 - self.paddle_height // 2
        self.ball_x: int = self.width // 2
        self.ball_y: int = self.height // 2
        self.ball_radius: int = 10
        self.ball_speed_x: int = 5
        self.ball_speed_y: int = 5
        self.score_left: int = 0
        self.score_right: int = 0

    def reset(self):
        self.paddle_left_y = self.height // 2 - self.paddle_height // 2
        self.paddle_right_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_speed_x = 5
        self.ball_speed_y = 5

    def step(self, action: Actions):
        self.screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if action == Actions.LEFT_UP:
            self.paddle_left_y -= self.paddle_speed
        elif action == Actions.LEFT_DOWN:
            self.paddle_left_y += self.paddle_speed
        elif action == Actions.RIGHT_UP:
            self.paddle_right_y -= self.paddle_speed
        elif action == Actions.RIGHT_DOWN:
            self.paddle_right_y += self.paddle_speed

        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (0, self.paddle_left_y, self.paddle_width, self.paddle_height),
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (
                self.width - self.paddle_width,
                self.paddle_right_y,
                self.paddle_width,
                self.paddle_height,
            ),
        )

        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        if self.ball_y - self.ball_radius:
            self.ball_speed_y *= -1
        if self.ball_y + self.ball_radius > self.height:
            self.ball_speed_y *= -1
        if (
            self.ball_x - self.ball_radius < self.paddle_width
            and self.paddle_left_y
            < self.ball_y
            < self.paddle_left_y + self.paddle_height
        ):
            self.ball_speed_x *= -1
        if (
            self.ball_x + self.ball_radius > self.width - self.paddle_width
            and self.paddle_right_y
            < self.ball_y
            < self.paddle_right_y + self.paddle_height
        ):
            self.ball_speed_x *= -1
        if self.ball_x < 0:
            self.score_right += 1
            self.reset()
        if self.ball_x > self.width:
            self.score_left += 1
            self.reset()

        pygame.draw.circle(
            self.screen, (255, 255, 255), (self.ball_x, self.ball_y), self.ball_radius
        )
        pygame.display.flip()
        self.clock.tick(GAME_SPEED)

    def render(self):
        pygame.display.flip()
        self.clock.tick(GAME_SPEED)

    def close(self):
        pygame.quit()

    def main(self):
        pygame.init()
        counter = 0
        while self.running:
            action = (counter + 1) % 4
            self.step(Actions(action))
            counter += 1
        self.close()

    def get_state(self) -> np.ndarray:
        return np.array(
            [self.paddle_left_y, self.paddle_right_y, self.ball_x, self.ball_y],
            dtype=np.int32,
        )
    
    def is_done(self) -> bool:
        return self.score_left == 10 or self.score_right == 10


if __name__ == "__main__":
    pong = Pong()
    pong.main()
