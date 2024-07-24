import curses
import sys
import time
from argparse import ArgumentParser
from itertools import count
from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt
from snake import Actions, Snake

sys.path.append(str(Path(__file__).parents[1]))
from utils.base_agent import BaseGameAgent
from utils.plotting import plot_scores

# WORLD_SIZE is the size of the world, which is a square grid.
# N_EPISODES is the number of episodes to train the agent.
WORLD_SIZE: int = 10
N_EPISODES = 10000


class SnakeGameAgent(BaseGameAgent):
    def __init__(self, render_episode: int = 100, render: bool = False):
        # Init snake.
        self.snake: Snake = Snake(world_size=WORLD_SIZE)
        self._render_episode = render_episode
        self._render = render
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Call the parent constructor.
        super().__init__(
            n_observations=len(self.snake.get_state()),
            n_actions=len(Actions),
            device=self._device,
            batch_size=128,
            gamma=0.999,
            eps_start=0.99,
            eps_end=0.001,
            eps_decay=500,
            tau=0.0001,
            lr=0.0001,
            memory_capacity=10000,
        )

        # Init curses.
        self.stdscr = curses.initscr()

    def main(self) -> None:
        top_score = 0
        scores: List[int] = []

        for i_episode in range(N_EPISODES):

            # Get state.
            self.snake.reset()
            state = self.snake.get_state()
            state = torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            # Counters.
            done = False
            previous_distance_to_food = self.snake.distance_to_food
            for t in count():

                # Render snake.
                if i_episode % self._render_episode == 0 and self._render:
                    self.stdscr.clear()
                    self.snake.render(self.stdscr)
                    self.stdscr.refresh()
                    time.sleep(0.1)

                # Get action.
                action = self.select_action(state)
                self.snake.set_direction(Actions(action.item()))

                # Move snake.
                self.snake.move()

                if self.snake.is_snake_eat_food():
                    self.snake.eat_food()
                    self.snake.generate_food()
                    reward = torch.tensor([0.999], device=self._device)
                elif self.snake.is_collision():
                    reward = torch.tensor([-0.999], device=self._device)
                    done = True
                else:
                    if self.snake.distance_to_food < previous_distance_to_food:
                        reward = torch.tensor([0.005], device=self._device)
                    else:
                        reward = torch.tensor([0.000], device=self._device)

                # Add a small positive reward for surviving each step.
                reward += torch.tensor([0.001], device=self._device)

                # Clamp the reward to be between -1 and 1.
                reward = torch.clamp(reward, -1.000, 1.000)

                # Get next state.
                next_state = torch.tensor(
                    self.snake.get_state(), dtype=torch.float32, device=self._device
                ).unsqueeze(0)

                # Update distance to food.
                previous_distance_to_food = self.snake.distance_to_food

                # Save transition.
                self.memory.push(state, action, next_state, reward)

                # Update state.
                state = next_state

                # Train DQN on a random batch.
                self.optimize_model()

                # Update target network.
                self.update_target_net()

                if done:
                    scores.append(self.snake.score)
                    plot_scores(scores)
                    if self.snake.score > top_score:
                        top_score = self.snake.score
                        print(
                            f"Episode {i_episode} finished after {t+1} timesteps with score {self.snake.score}."
                        )
                    break

        # Plot scores.
        plot_scores(scores, show_result=True)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--render_episode",
        type=int,
        default=100,
        help="Render every nth episode. Set to 0 to disable rendering.",
    )
    arg_parser.add_argument("--render", action="store_true", help="Render the game.")
    args, unknown = arg_parser.parse_known_args()
    agent = SnakeGameAgent(render_episode=args.render_episode, render=args.render)
    agent.main()
