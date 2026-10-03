import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
torch.set_num_threads(1)

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DDPG import DDPG
from TD3 import TD3
from helper import rolling_window, episode_reward_plot


def test_rolling_windows_use_step_between_windows():
    np.testing.assert_array_equal(
        rolling_window(np.arange(8), 3, 2), [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    )


@pytest.mark.parametrize("rewards", [[], [1.0], [1.0, 2.0, 3.0, 4.0]])
def test_plot_accepts_short_runs(rewards, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    episode_reward_plot(rewards, 1, window_size=7)


@pytest.mark.parametrize("agent_type", [DDPG, TD3])
def test_learning_updates_parameters_and_evaluation_is_separate(agent_type):
    env = gym.make("LunarLander-v3", continuous=True, max_episode_steps=8)
    evaluation = gym.make("LunarLander-v3", continuous=True, max_episode_steps=8)
    try:
        agent = agent_type(env, batch_size=4, replay_size=32, eval_env=evaluation)
        initial = [parameter.detach().clone() for parameter in agent.Actor.parameters()]
        rewards, evaluations = agent.learn(16, seed=23, plot=False)
        assert len(rewards) == len(evaluations) == 2
        assert np.isfinite(rewards + evaluations).all()
        assert any(not torch.equal(a, b) for a, b in zip(initial, agent.Actor.parameters()))
        env.reset(seed=11)
        lander = env.unwrapped.lander
        state = tuple(lander.position) + tuple(lander.linearVelocity) + (lander.angle, lander.angularVelocity)
        agent.eval_episodes(n=1)
        current = tuple(lander.position) + tuple(lander.linearVelocity) + (lander.angle, lander.angularVelocity)
        np.testing.assert_array_equal(current, state)
        assert agent.steps_completed == 16
        assert agent.evaluation_steps == [8, 16]
    finally:
        env.close()
        evaluation.close()


@pytest.mark.parametrize("agent_type", [DDPG, TD3])
def test_single_training_step_completes(agent_type, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = gym.make("LunarLander-v3", continuous=True, max_episode_steps=8)
    try:
        agent = agent_type(env, batch_size=4, replay_size=32)
        agent.learn(1, seed=5, plot=True)
        assert agent.steps_completed == 1
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["ddpg", "td3"])
def test_runner_exports_seeded_metrics_and_loadable_actor(algorithm, tmp_path):
    root = Path(__file__).resolve().parents[1]
    first, second = tmp_path / "first", tmp_path / "second"
    for destination in [first, second]:
        result = subprocess.run(
            [sys.executable, str(root / "train.py"), "--algorithm", algorithm,
             "--steps", "16", "--batch-size", "4", "--seed", "17",
             "--max-episode-steps", "8", "--output", str(destination)],
            capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (first / "evaluations.csv").read_text() == (second / "evaluations.csv").read_text()
    assert (first / "training.csv").read_text() == (second / "training.csv").read_text()
    config = json.loads((first / "config.json").read_text())
    assert config["seed"] == 17
    assert config["steps_completed"] == 16
    rows = list(csv.DictReader((first / "evaluations.csv").open()))
    assert [int(row["training_step"]) for row in rows] == [8, 16]
    from ActorCriticNetworks import ActorNetwork
    actor = ActorNetwork(8, 2)
    actor.load_state_dict(torch.load(first / "actor.pt", map_location=actor.device, weights_only=True))
    action = actor(torch.zeros(1, 8, device=actor.device)).detach().cpu().numpy()
    assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
