import numpy as np
import matplotlib.pyplot as plt
import math
import time


def rolling_window(a, window, step_size):
    """Create a rolling window view of a numpy array."""

    if window < 1 or step_size < 1:
        raise ValueError("window and step_size must be positive")
    return np.lib.stride_tricks.sliding_window_view(a, window)[::step_size]


ax = None
fig = None

def episode_reward_plot(rewards, frame_idx, window_size=5, step_size=1, updating=False):
    """Plot episode rewards rolling window mean, min-max range and standard deviation.

    Parameters
    ----------
    rewards : list
        List of episode rewards.
    frame_idx : int
        Current frame index.
    window_size : int
        Rolling window size.
    step_size: int
        Step size between windows.
    updating: bool
        You can try to set updating to True, which hinders matplotlib to create a new window for every plot.
        Doesn't work with my Pycharm SciView currently.
    """
    global ax
    global fig

    if not len(rewards):
        return
    window_size = min(window_size, len(rewards))
    rewards_rolling = rolling_window(np.array(rewards), window_size, step_size)
    mean = np.mean(rewards_rolling, axis=1)
    std = np.std(rewards_rolling, axis=1)
    minimum = np.min(rewards_rolling, axis=1)
    maximum = np.max(rewards_rolling, axis=1)
    x = np.arange(len(mean)) * step_size + (window_size - 1) / 2

    if ax is None: #or not updating:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.clear()
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, facecolor='blue')
    ax.fill_between(x, minimum, maximum, alpha=0.1, facecolor='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    if updating:
        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.01)
        # plt.ion()
        # plt.show()
        # plt.clf()
    else:
        plt.show(block=False)
        plt.savefig('rewards.png')




def visualize_agent(env, agent, timesteps=500):
    """ Visualize an agent performing inside a Gym environment. """
    obs, _ = env.reset()
    for timestep in range(1, timesteps + 1):
        env.render()
        action = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        # 30 FPS
        time.sleep(0.033)

def video_agent(env, agent, n_episodes=500):
    """ Visualize an agent performing inside a Gym environment. """
    ce = 0
    obs, _ = env.reset()
    while True:
        env.render()
        action = agent.choose_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            ce=ce+1
            if ce>=n_episodes:
                break
    env.close()

