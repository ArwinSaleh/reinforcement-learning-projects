import random
import sys
import time
from math import sqrt, exp
from pathlib import Path
from typing import Any, List
import torch
from snake import Snake, Actions
import curses
from torch import optim

sys.path.append(str(Path(__file__).parents[1]))
from utils.dqn import DQN
from utils.q_trainer import QTrainer
from utils.replay_memory import ReplayMemory, Transition

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# WORLD_SIZE is the size of the world, which is a square grid
# MEMORY_CAPACITY is the maximum capacity of the replay buffer
# MAX_REWARD is the maximum reward that can be obtained
WORLD_SIZE: int = 5
MEMORY_CAPACITY: int = 100000
MAX_REWARD = 100000


class SnakeGameAgent:
    def __init__(self):
        self.steps_done = 0

        # Init snake
        self.snake: Snake = Snake(world_size=WORLD_SIZE)

        # Init DQN
        n_observations = 11
        n_actions = len(Actions)
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory: ReplayMemory = ReplayMemory(MEMORY_CAPACITY)

        self.n_gens = 0
        #self.stdscr = curses.initscr()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            random_action = random.choice(range(len(Actions)))
            return torch.tensor([[random_action]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def snake_game(self) -> None:
        # Get state
        state = self.snake.get_state()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        snake_alive_counter = 0.0
        consecutive_foods = 0
        while True:
            print(f"Generation: {self.n_gens}, Score: {self.snake.score}, Duration: {int(snake_alive_counter)}")
            
            # Render snake
            #self.stdscr.clear()
            #self.snake.render(self.stdscr)
            #self.stdscr.refresh()
            #time.sleep(0.1)

            # Get action
            action = torch.tensor(self.select_action(state))
            self.snake.set_direction(Actions(action.item()))

            # Move snake
            self.snake.move()
            
            # Check if snake ate apple
            if self.snake.is_snake_eat_food():
                self.snake.eat_food()
                self.snake.generate_food()
                consecutive_foods += 1

            # Get reward - distance to apple
            if not self.snake.is_collision():
                snake_alive_counter += 1.0
                reward = torch.tensor([snake_alive_counter + consecutive_foods * 100.0])
            else:
                snake_alive_counter = 0.0
                reward = torch.tensor([-100.0])
                self.snake.reset()
                self.n_gens += 1
            
            # Clip reward
            #reward = torch.clamp(reward, -1.0, 1.0)
                
            # Get next state
            next_state = torch.tensor(self.snake.get_state(), dtype=torch.float32).unsqueeze(0)

            # Save transition
            self.memory.push(
                state, action, next_state, reward.unsqueeze(0)
            )
            
            # Update state
            state = next_state

            # Train DQN on a random batch
            self.optimize_model()
            
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_state_dict)


if __name__ == "__main__":
    agent = SnakeGameAgent()
    agent.snake_game()
