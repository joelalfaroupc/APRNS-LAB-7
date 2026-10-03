"""Render the archived DDPG/TD3 evaluations without altering the source CSVs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_results(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def analyze(output=None):
    output = Path(output) if output else ROOT / "assets"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "lunar-lander-portfolio"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    summary = {}
    for axis, (algorithm, color) in zip(axes, [("DDPG", "#2369b5"), ("TD3", "#008878")]):
        data = read_results(ROOT / f"{algorithm.lower()}.csv")
        rewards, rolling = data["Episode Reward"], data["Mean R"]
        x = data["Episode"] + 1
        axis.axhline(0, color="#9aa8b7", linewidth=0.7)
        axis.plot(x, rewards, color=color, alpha=0.24, linewidth=1,
                  label="Evaluation mean (3 episodes, rounded)")
        axis.plot(x, rolling, color=color, linewidth=2.5,
                  label="Rolling mean (up to 100 evaluations)")
        axis.set_title(f"{algorithm}  ·  {len(rewards)} evaluations", loc="left", fontweight="bold")
        axis.set_xlabel("Evaluation index")
        axis.grid(axis="y", alpha=0.15)
        axis.legend(loc="lower right", fontsize=8.5, frameon=False)
        summary[algorithm] = {
            "evaluations": len(rewards), "final_rolling_mean_100": float(rolling[-1]),
            "last_20_mean_of_rounded_evaluations": float(rewards[-20:].mean()),
            "best_rounded_evaluation_mean": float(rewards.max()),
            "final_rounded_evaluation_mean": float(rewards[-1]),
        }
    axes[0].set_ylabel("Episode return")
    fig.suptitle("Continuous lunar landing · archived learning curves", x=0.07,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.07, 0.02, "One archived run per algorithm · seeds not recorded · evaluation indices are not training steps",
             color="#526171", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.93))
    fig.savefig(output / "learning-curves.svg", metadata={"Date": None})
    fig.savefig(output / "learning-curves.png", dpi=170)
    plt.close(fig)
    (output / "archived-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
