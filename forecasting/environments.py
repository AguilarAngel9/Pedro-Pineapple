# Environment for Stock or Index prediction.
# Author: @THEFFTKID.

from enum import Enum
from typing import Union, Tuple, Dict, Any, Literal

import matplotlib.pyplot as plt
import gymnasium as gym
import features as ft
import pandas as pd
import numpy as np


class Actions(Enum):
    """
    Discrete set of actions for the agent (stock/index).
    """
    down_movement = 0
    no_movement = 1
    up_movement = 2


class Forecasting(gym.Env):
    """
    Stock - Index forecasting environment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        features_list: list[Literal[
            'open',
            'high',
            'low',
            'n10_rolling_mean',
            'n10_weighted_rolling_mean',
            'momentum',
            'close'
            'nday_tendency_removal'
            ]]
    ) -> None:
        assert df.ndim == 2

        # Raw index - stock data.
        self.df = df
        # Size of the horizon.
        self.window_size = window_size

        # Features (phi).
        self.features_list = features_list
        self.prices, self.signal_features = self._process_data()
        # Shape of the observation space
        self.shape = (window_size, self.signal_features.shape[1])

        # Action space.

        # Finite set of actions A = {1, ..., n}.
        self.action_space = gym.spaces.Discrete(n=len(Actions), start=0)
        INF = 1e10
        # Space for continous vectors R^{n}.
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32
        )

        # Episode

        # Initial position over the time series.
        self._start_tick = self.window_size
        # Final position over the time series.
        self._end_tick = len(self.prices) - 1
        # Truncated flag.
        self._truncated = None
        # Current position over the time series.
        self._current_tick = None
        # Total accumulated reward over time G(t).
        self._total_reward = None
        # History of the environment.
        self.history = None
        # Trajectory of actions.
        self.actions_history = None
        

    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None
    ) -> Tuple[Any, Union[Dict[str, Any], None]]:
        """
        Resets the environment to an initial state
        , required before calling step.
        Returns the first agent observation for an episode and information.
        """
        super().reset(seed=seed, options=options)

        self.action_space.seed(
            int((self.np_random.uniform(0, seed if seed is not None else 1)))
        )

        # Reset truncated flag.
        self._truncated = False
        # Reset to the first tick used.
        self._current_tick = self._start_tick
        # Reset total reward to zero.
        self._total_reward = 0.
        # Reset history.
        self.history = {}
        # Reset actions history.
        self.actions_history = []

        # Get the first state.
        observation = self._get_observation()
        # Update first step.
        info = self._get_info()

        return observation, info

    def step(self, action: Union[int, np.int64]):
        """
        Updates an environment with actions returning the next
        agent observation, the reward for taking that actions,
        if the environment has terminatedor truncated due to the latest
        action and information from the environment about the step.
        """
        # Append current action to actions history.
        self.actions_history.append(action)

        # Set truncated to false.
        self._truncated = False

        # Add one step to the last step.
        self._current_tick += 1

        # Check if the last step.
        if self._current_tick == self._end_tick:
            self._truncated = True

        # Calculate instant reward.
        step_reward = self._calculate_reward(action)

        # Update the total reward using the current reward.
        self._total_reward += step_reward

        observation = self._get_observation()

        # Get the current state of the environment.
        info = self._get_info()

        # Update the history.
        self._update_history(info)

        return observation, step_reward, False, self._truncated, info

    def _get_info(self) -> Dict[str, float]:
        """
        Creates a dict entry with the current state of the environment.
        """
        return dict(
            total_reward=self._total_reward
        )

    def _get_observation(self) -> np.ndarray:
        """
        Return the next state vector phi_{t + 1}.
        """
        init_slice = self._current_tick - self.window_size + 1
        end_slice = self._current_tick + 1
        return self.signal_features[init_slice:end_slice]

    def _update_history(self, info) -> None:
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self, title: str = None):
        """
        Plot all elements of the environment.
        """
        window_ticks = np.arange(len(self.actions_history))

        # Plot prices time series.
        plt.plot(self.prices, 'b')

        # Create xticks for each action vector.
        up_ticks = []
        no_ticks = []
        down_ticks = []

        for tick in window_ticks:
            if self.actions_history[tick] == Actions.up_movement.value:
                up_ticks.append(tick)
            elif self.actions_history[tick] == Actions.no_movement.value:
                no_ticks.append(tick)
            elif self.actions_history[tick] == Actions.down_movement.value:
                down_ticks.append(tick)

        plt.plot(
            up_ticks, self.prices[up_ticks], 'g^'
        )
        plt.plot(
            down_ticks, self.prices[down_ticks], 'rv'
        )
        plt.plot(
            no_ticks, self.prices[no_ticks], 'y>'
        )

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
        )
        plt.grid()

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def _process_data(self) -> Tuple[np.array, np.array]:
        """
        Create all feature vectors phi_{s}.
        """
        # Deep copy of data.
        data = self.df.copy(deep=True)

        data.reset_index(inplace=True, drop=True)

        # Standardize columns.
        data.columns = list(map(str.lower, data.columns))

        # 10-day rolling mean.
        data['n10_rolling_mean'] = ft.n_rolling_mean(
            x_=data['close'], n=10
        )
        # 10-day weighted  rolling mean.
        data['n10_weighted_rolling_mean'] = ft.n_rolling_mean(
            x_=data['close'], n=10
        )
        # Momentum.
        data['momentum'] = ft.momentum(x_=data['close'], n=10)

        # Removal of tendency
        data['nday_tendency_removal'] = ft.tendency_removal(df_close = data['close'], n = 10)

        features = data[
            self.features_list
            # [
            #     'open',
            #     'high',
            #     'low',
            #     'n10_rolling_mean',
            #     'n10_weighted_rolling_mean',
            #     'momentum',
            #     'close'
            #     'nday_tendency_removal'
            # ]
        ].to_numpy()

        prices = data['close'].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)

        signal_features = np.concatenate(
            (features, diff.reshape(-1, 1)), axis=1
        )

        return prices, signal_features

    def _calculate_reward(self, action):
        """
        Ad Hoc reward function for index/stock.
        """
        # All steps starts with a base reward of 0.
        step_reward = 0

        # Current price.
        current_price = self.prices[self._current_tick]

        # Last price.
        last_trade_price = self.prices[self._current_tick - 1]

        # TD(0) difference.
        price_diff = current_price - last_trade_price

        # Relative change in price since the previous time step.
        relative_change = 100 * (price_diff / last_trade_price)

        if (
            action == Actions.up_movement.value and relative_change >= 0.1
        ):
            step_reward = price_diff
        elif (
            action == Actions.down_movement.value and relative_change <= -0.1
        ):
            step_reward = price_diff
        elif (
            action == Actions.no_movement and abs(relative_change) < 0.1
        ):
            step_reward = price_diff

        return abs(step_reward)
