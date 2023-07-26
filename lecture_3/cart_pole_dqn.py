# %% [markdown]
# # Cart Pole DQN Solution
# 
# ![cart_pole](../images/lecture_3/cart_pole.png)
# 

# %% [markdown]
# ## Gym Description
# 
# ### Description
# 
# This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
# ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
# The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
# in the left and right direction on the cart.
# 
# ### Action Space
# 
# The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
# of the fixed force the cart is pushed with.
# 
# - 0: Push cart to the left
# - 1: Push cart to the right
# 
# **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
# the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
# 
# ### Observation Space
# 
# The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
# 
# | Num | Observation           | Min                 | Max               |
# | --- | --------------------- | ------------------- | ----------------- |
# | 0   | Cart Position         | -4.8                | 4.8               |
# | 1   | Cart Velocity         | -Inf                | Inf               |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |
# 
# **Note:** While the ranges above denote the possible values for observation space of each element,
# it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
# 
# - The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
#   if the cart leaves the `(-2.4, 2.4)` range.
# - The pole angle can be observed between `(-.418, .418)` radians (or **±24°**), but the episode terminates
#   if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
# 
# ### Rewards
# 
# Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
# including the termination step, is allotted. The threshold for rewards is 475 for v1.
# 
# ### Starting State
# 
# All observations are assigned a uniformly random value in `(-0.05, 0.05)`
# 
# ### Episode End
# 
# The episode ends if any one of the following occurs:
# 
# 1. Termination: Pole Angle is greater than ±12°
# 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
# 3. Truncation: Episode length is greater than 500 (200 for v0)
# 
# ### Arguments
# 
# ```python
# import gymnasium as gym
# gym.make('CartPole-v1')
# ```
# 
# On reset, the `options` parameter allows the user to change the bounds used to determine
# the new random state.
# """
# 

# %% [markdown]
# ## Import Env
# 

# %%
import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
for k, v in gym.envs.registry.items():
    print(str(k)+"\t"+str(v))


# %%
env = gym.make("CartPole-v1")
print("Observation space: ", env.observation_space)
print("Observation shape: ",env.observation_space.shape)
print("Action space: ", env.action_space)

# %% [markdown]
# ## Transition
# 

# %%
from collections import namedtuple
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward"))


# %% [markdown]
# ## Experience
# 

# %%
from collections import deque
import random
from typing import List


class Experience:
    def __init__(self, maxlen: int) -> None:
        # maxlen: if the deque is full, stored items will pop from the head
        self.experience = deque([], maxlen=maxlen)

    def append(self, transition: Transition) -> None:
        self.experience.append(transition)

    def sample(self, batch_size) -> List:
        return random.sample(self.experience, batch_size)

    def get_length(self) -> int:
        return len(self.experience)


# %% [markdown]
# ## Q network
# 

# %%

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_observations: int, num_actions: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(num_observations, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, num_actions)
        self.selu = nn.SELU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q_actions = self.linear_1(state)
        q_actions = self.selu(q_actions)
        q_actions = self.linear_2(q_actions)
        q_actions = self.selu(q_actions)
        q_actions = self.linear_3(q_actions)
        return q_actions


# %% [markdown]
# ## Epsilon-greedy

# %%
import numpy as np


def epsilon_greedy(q_approximator: nn.Module, state: torch.Tensor, num_actions: int, epsilon: float):
    q_actions = q_approximator(state).detach().cpu().numpy()
    probability = np.ones(num_actions)*epsilon/num_actions
    idx_action_with_max_q = np.argmax(q_actions)
    probability[idx_action_with_max_q] = 1-np.sum(probability[1:])
    return np.random.choice(np.arange(num_actions), p=probability)


epsilon_greedy(DQN(env.observation_space.shape[0], env.action_space.n), torch.tensor(
    env.reset()[0]), env.action_space.n, 0.1)


# %% [markdown]
# ## Update DQN

# %%
def update_dqn(q_approximator:DQN,experience:Experience,batch_size:int,gamma:float,loss_fn:nn.Module,optimizer:torch.optim.Optimizer,device:str):
    batch=experience.sample(batch_size)
    batch_state=torch.tensor([t.state for t in batch],device=device)
    batch_action=torch.tensor([t.action for t in batch],device=device).unsqueeze(-1)
    batch_next_state=torch.tensor([t.next_state for t in batch],device=device)
    batch_reward=torch.tensor([t.reward for t in batch],device=device).unsqueeze(-1)
    q_state_action=q_approximator(batch_state).gather(1,batch_action)
    q_next_state_action=torch.zeros(batch_action.size(),device=device)
    with torch.no_grad():
        q_next_state_action=torch.max(q_approximator(batch_next_state),1)[0].unsqueeze(-1)
    q_target=batch_reward+gamma*q_next_state_action
    optimizer.zero_grad()
    loss=loss_fn(q_target,q_state_action)
    loss.backward()
    optimizer.step()
    

# %% [markdown]
# ## Test DQN

# %%
def test_dqn(env:gym.Env,q_approximator:DQN,device:str):
    duration=0
    state=env.reset()[0]
    is_terminated=False
    while not is_terminated:
        qs_state_action=q_approximator(torch.tensor(state,device=device))
        action=torch.argmax(qs_state_action).detach().cpu().item()
        next_state,reward,is_terminated,is_truncated,info=env.step(action)
        state=next_state
        duration+=1
    print("duration: ",duration)
        

# %% [markdown]
# ## Q-learning

# %%
def q_learning(env: gym.Env, num_episodes: int, batch_size: int = 64, alpha: float = 1e-3, gamma: float = 0.9, epsilon: float = 0.1, device: str = "cpu", print_step=100):

    q_approximator = DQN(
        env.observation_space.shape[0], env.action_space.n).to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(q_approximator.parameters(), lr=alpha)

    experience = Experience(maxlen=256)

    for episode_i in range(num_episodes):
        state = env.reset()[0]
        is_terminated = False
        while not is_terminated:
            action = epsilon_greedy(q_approximator, torch.tensor(
                state, device=device), env.action_space.n, epsilon)
            next_state, reward, is_terminated, is_truncated, info = env.step(
                action)
            experience.append(Transition(state, action, next_state, reward))
            state = next_state

            if experience.get_length() >= batch_size:
                update_dqn(q_approximator, experience, batch_size,
                           gamma, loss_fn, optimizer, device)

        if episode_i % print_step == 0:
            print("-------- episode %d ---------" % episode_i)
            test_dqn(env, q_approximator, device)


# %%
q_learning(env, int(1e4),device= "cpu")



