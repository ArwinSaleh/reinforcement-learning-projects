import sys
from itertools import count
from pathlib import Path
from typing import List

import torch
from pong import Actions, Pong

sys.path.append(str(Path(__file__).parents[1]))
from utils.base_agent import BaseGameAgent
from utils.plotting import plot_scores

# N_EPISODES is the number of episodes to train the agent.
N_EPISODES = 10000


class PongGameAgent(BaseGameAgent):
    def __init__(self):
        # Init pong.
        self.pong = Pong()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Call the parent constructor.
        super().__init__(
            n_observations=len(self.pong.get_state()),
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

    def main(self) -> None:
        episode_durations: List[int] = []

        for i_episode in range(N_EPISODES):

            # Get state.
            self.pong.reset()
            state = self.pong.get_state()
            state = torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            for t in count():
                # Select and perform an action.
                action = self.select_action(state)
                self.pong.step(Actions(action.item()))

                # Observe new state.
                if not self.pong.is_done:
                    reward = 0.001
                    next_state = self.pong.get_state()
                    next_state = torch.tensor(
                        next_state, dtype=torch.float32, device=self._device
                    ).unsqueeze(0)
                else:
                    reward = -1.0
                    next_state = None

                # Store the transition in memory.
                self.memory.push(
                    state,
                    action,
                    next_state,
                    torch.tensor([reward], device=self._device),
                )

                # Move to the next state.
                state = next_state

                # Perform one step of the optimization (on the target network).
                self.optimize_model()
                
                # Update the target network.
                self.update_target_net()

                if self.pong.is_done:
                    # Save the duration of the current episode.
                    episode_durations.append(t + 1)
                    plot_scores(episode_durations, show_result=False)
                    break

if __name__ == "__main__":
    agent = PongGameAgent()
    agent.main()
    plot_scores(agent.scores, show_result=True)