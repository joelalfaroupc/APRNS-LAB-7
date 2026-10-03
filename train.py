"""Seeded training and artifact export for continuous LunarLander-v3."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import gymnasium as gym
import numpy as np
import torch
from DDPG import DDPG
from TD3 import TD3


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def record_policy(agent, output, episodes, seed, max_episode_steps):
    env = gym.wrappers.RecordVideo(
        gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array",
                 max_episode_steps=max_episode_steps),
        str(output / "videos"), episode_trigger=lambda _: True,
        name_prefix="final-policy", disable_logger=True,
    )
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 20000 + episode)
            while True:
                observation, _, terminated, truncated, _ = env.step(agent.choose_action(observation))
                if terminated or truncated:
                    break
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ddpg", "td3"], default="td3")
    parser.add_argument("--steps", type=positive_int, default=500000,
                        help="Maximum training steps; the original early-stop rule still applies")
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--max-episode-steps", type=positive_int, default=1000)
    parser.add_argument("--video-episodes", type=nonnegative_int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / f"{args.algorithm}-seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    batch_size = args.batch_size or (64 if args.algorithm == "ddpg" else 100)
    env = gym.make("LunarLander-v3", continuous=True, max_episode_steps=args.max_episode_steps)
    evaluation = gym.make("LunarLander-v3", continuous=True, max_episode_steps=args.max_episode_steps)
    try:
        agent_type = DDPG if args.algorithm == "ddpg" else TD3
        agent = agent_type(env, batch_size=batch_size, eval_env=evaluation)
        rewards, evaluations = agent.learn(args.steps, seed=args.seed, plot=False)
        write_csv(output / "training.csv", ["episode", "training_step", "episode_return"],
                  ((i, step, reward) for i, (step, reward) in
                   enumerate(zip(agent.evaluation_steps, rewards), start=1)))
        write_csv(output / "evaluations.csv",
                  ["evaluation", "training_step", "mean_return", "rolling_mean_100"],
                  ((i + 1, step, reward, float(np.mean(evaluations[max(0, i - 99):i + 1])))
                   for i, (step, reward) in enumerate(zip(agent.evaluation_steps, evaluations))))
        torch.save(agent.Actor.state_dict(), output / "actor.pt")
        critics = {"critic": agent.Critic} if args.algorithm == "ddpg" else {
            "critic1": agent.Critic1, "critic2": agent.Critic2}
        for name, critic in critics.items():
            torch.save(critic.state_dict(), output / f"{name}.pt")
        config = {
            "algorithm": args.algorithm, "environment": "LunarLander-v3", "continuous": True,
            "seed": args.seed, "evaluation_seed": args.seed + 10000,
            "steps_requested": args.steps, "steps_completed": agent.steps_completed,
            "max_episode_steps": args.max_episode_steps, "batch_size": batch_size,
            "gamma": agent.gamma, "replay_size": 1000000, "evaluation_episodes": 3,
            "actor_hidden_units": [400, 300], "critic_hidden_units": [400, 300],
            "actor_lr": 0.0001 if args.algorithm == "ddpg" else 0.001,
            "critic_lr": 0.001, "critic_weight_decay": 0.01 if args.algorithm == "ddpg" else 0,
            "tau": 0.001 if args.algorithm == "ddpg" else 0.005,
            "early_stop": "more than 10 evaluations and last-five mean > 220",
            "device": str(agent.Actor.device), "torch_threads": 1,
            "versions": {"torch": torch.__version__, "gymnasium": gym.__version__, "numpy": np.__version__},
            "weights": "final state dictionaries; not resumable training checkpoints",
        }
        if args.algorithm == "td3":
            config.update(policy_noise=agent.policy_noise, noise_clip=agent.noise_clip,
                          policy_freq=agent.policy_freq, exploration_noise="scalar Gaussian, sigma=0.1")
        else:
            config.update(exploration_noise="Ornstein–Uhlenbeck, theta=0.2, sigma=0.15, dt=0.01")
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if args.video_episodes:
            record_policy(agent, output, args.video_episodes, args.seed, args.max_episode_steps)
        print(f"Saved {agent.steps_completed} training steps and {len(evaluations)} evaluations to {output}")
    finally:
        evaluation.close()
        env.close()


if __name__ == "__main__":
    main()
