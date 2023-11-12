# Custom environment for Stock prediction.
# Author: @THEFFTKID.

from time import time
from enum import Enum

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class Actions(Enum):
    """
    Discrete set of actions for the agent (stock/index).
    """
    up_movement = 1
    no_movement = 0
    down_movement = -1

# TODO: Delete this shit.
# class Positions(Enum):
#     Short = 0
#     Long = 1

#     def opposite(self):
#         return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # Action space.
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # Episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        # Ad hoc needs.
        self._actions_history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        # self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self._actions_history = []

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        # Add action.
        self._actions_history.append(action)

        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            print('Entré')
            self._truncated = True

        # Calculate instant reward.
        step_reward = self._calculate_reward(action)
        # Update the total reward using the current reward.
        self._total_reward += step_reward

        # self._update_profit(action)

        # trade = False
        # if (
        #     (action == Actions.up_movement.value and self._position == Positions.Short) or
        #     (action == Actions.down_movement.value and self._position == Positions.Long)
        # ):
        #     trade = True

        # if trade:
        #     self._position = self._position.opposite()
        #     self._last_trade_tick = self._current_tick

        # self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()
        

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            # position=self._position
        )

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            pass
            # color = None
            # if position == Positions.Short:
            #     color = 'red'
            # elif position == Positions.Long:
            #     color = 'green'
            # if color:
            #     plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._actions_history))
        plt.plot(self.prices)

        up_ticks = []
        no_ticks = []
        down_ticks = []
    
        for i, tick in enumerate(window_ticks):
            
            print(Actions.up_movement.value, self._actions_history[i])
            
            if self._actions_history[i] == Actions.up_movement.value:
                up_ticks.append(tick)
            elif self._actions_history[i] == Actions.no_movement.value:
                no_ticks.append(tick)
            elif self._actions_history[i] == Actions.down_movement.value:
                down_ticks.append(tick)

        plt.plot(up_ticks, self.prices[up_ticks], 'ro')
        plt.plot(no_ticks, self.prices[no_ticks], 'go')
        # plt.plot(down_ticks, self.prices[down_ticks], 'bo')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            # + ' ~ ' +
            # "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self) -> np.array:
        """
        Create the features to use.
        """
        prices = self.df.loc[:, 'Close'].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)

        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        """
        Ad Hoc reward function for index/stock.
        """
        # All steps starts with a base reward of 0.
        step_reward = 0

        # Current price.
        current_price = self.prices[self._current_tick]

        # Last prices
        # last_trade_price = self.prices[self._last_trade_tick]
        last_trade_price = self.prices[self._current_tick - 1]

        print(current_price, last_trade_price)
        
        # TD(0) difference.
        price_diff = current_price - last_trade_price

        # Relative change in price since the previous time step.
        relative_change =  100 * (price_diff / last_trade_price)

        print(relative_change)
        # TODO: Propose new threshold based on data.
        # TODO: Check if log_{10} or ln.
        # step_reward = np.log((current_price / last_trade_price))
        # TODO: Implement Sharpe ratio.
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

    def _update_profit(self, action):
        """
        Apparently we do no need profit.
        """
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        return price_diff

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
