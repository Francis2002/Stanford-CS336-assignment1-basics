"""
Plot ablation results from metrics.jsonl files.
Usage: python scripts/plot_results.py
"""
import json
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip install matplotlib")
    print("Falling back to text summary only.")


def load_run(run_dir):
    """Load metrics and config from a run directory."""
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    
    metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                metrics.append(json.loads(line))
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    
    return metrics, config


def summarize_run(metrics, config):
    """Get summary stats for a run."""
    if not metrics:
        return None
    
    final_val_loss = metrics[-1].get("val_loss", float("inf"))
    min_val_loss = min(m.get("val_loss", float("inf")) for m in metrics)
    total_steps = metrics[-1].get("step", 0)
    wall_time = metrics[-1].get("wall_time", 0)
    
    return {
        "final_val_loss": final_val_loss,
        "min_val_loss": min_val_loss,
        "total_steps": total_steps,
        "wall_time_min": wall_time / 60,
    }


def make_label(config):
    """Create a short label from config."""
    parts = []
    lr = config.get("learning_rate_base", "?")
    bs = config.get("batch_size", "?")
    d = config.get("d_model", "?")
    layers = config.get("num_layers", "?")
    warmup = config.get("steps_for_warmup", "?")
    ctx = config.get("context_length", "?")
    parts.append(f"lr={lr}")
    parts.append(f"bs={bs}")
    parts.append(f"d={d}")
    parts.append(f"L={layers}")
    if warmup != 500:  # non-default
        parts.append(f"wu={warmup}")
    if ctx != 256:  # non-default
        parts.append(f"ctx={ctx}")
    return ", ".join(parts)


def main():
    base_dir = Path(__file__).resolve().parent.parent / "cs336_basics" / "checkpoints"
    
    if not base_dir.exists():
        print(f"No checkpoints directory found at {base_dir}")
        return
    
    runs = []
    for run_dir in sorted(base_dir.iterdir()):
        if run_dir.is_dir():
            metrics, config = load_run(str(run_dir))
            if metrics:
                summary = summarize_run(metrics, config)
                if summary:
                    runs.append({
                        "name": run_dir.name,
                        "metrics": metrics,
                        "config": config,
                        "summary": summary,
                    })
    
    if not runs:
        print("No completed runs found.")
        return
    
    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Run':<25} {'Config':<40} {'Min Val Loss':<12} {'Time (min)':<10}")
    print("-" * 80)
    
    for run in sorted(runs, key=lambda r: r["summary"]["min_val_loss"]):
        label = make_label(run["config"]) if run["config"] else run["name"]
        s = run["summary"]
        print(f"{run['name']:<25} {label:<40} {s['min_val_loss']:<12.4f} {s['wall_time_min']:<10.1f}")
    
    print("=" * 80)
    
    if not HAS_MPL:
        return
    
    # Plot validation loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for run in runs:
        steps = [m["step"] for m in run["metrics"]]
        val_losses = [m["val_loss"] for m in run["metrics"]]
        wall_times = [m["wall_time"] / 60 for m in run["metrics"]]
        label = make_label(run["config"]) if run["config"] else run["name"]
        
        # Smooth for readability (rolling average)
        window = max(1, len(val_losses) // 100)
        if len(val_losses) > window:
            smoothed = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            steps_s = steps[window-1:]
            times_s = wall_times[window-1:]
        else:
            smoothed = val_losses
            steps_s = steps
            times_s = wall_times
        
        axes[0].plot(steps_s, smoothed, label=label, alpha=0.8)
        axes[1].plot(times_s, smoothed, label=label, alpha=0.8)
    
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Val Loss vs Steps")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Wall Time (minutes)")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Val Loss vs Wall Time")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "ablation_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_dir / 'ablation_results.png'}")


if __name__ == "__main__":
    main()
