import sys
from itertools import count
from pathlib import Path
from typing import List

import gymnasium as gym
import torch
from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).parents[1]))
from utils.base_agent import BaseGameAgent
from utils.plotting import plot_scores

if torch.cuda.is_available():
    N_EPISODES = 300
else:
    N_EPISODES = 50


class CartPoleAgent(BaseGameAgent):
    def __init__(self, render_episode: int = 100, render: bool = False):
        # Init environment.
        self.env = gym.make("CartPole-v0")
        self._render_episode = render_episode
        self._render = render
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Call the parent constructor.
        super().__init__(
            n_observations=self.env.observation_space.shape[0],
            n_actions=self.env.action_space.n,
            device=self._device,
        )

    def main(self) -> None:
        episode_durations: List[int] = []

        for i_episode in range(N_EPISODES):

            # Initialize the environment and get its state.
            state, info = self.env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                reward = torch.tensor([reward], device=self._device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self._device
                    ).unsqueeze(0)

                # Store the transition in memory.
                self.memory.push(state, action, next_state, reward)

                # Move to the next state.
                state = next_state

                # Perform one step of the optimization (on the policy network).
                self.optimize_model()

                # Soft update of the target network's weights
                self.update_target_net()

                if done:
                    episode_durations.append(t + 1)
                    plot_scores(episode_durations)
                    break

                # Render environment.
                if i_episode % self._render_episode == 0 and self._render:
                    self.env.render()

        print("Complete")
        plot_scores(episode_durations, show_result=True)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    agent = CartPoleAgent(render=True)
    agent.main()
    agent.env.close()
