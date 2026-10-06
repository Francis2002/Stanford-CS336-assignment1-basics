"""
Build every figure referenced by REPORT.md.

Usage:
    uv run python scripts/make_plots.py                 # all figures
    uv run python scripts/make_plots.py tinystories_lr   # one or more by name
    uv run python scripts/make_plots.py --list
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

SYSTEM_STAGES = [
    ("eager\nFP32", ["sys_00_eager_fp32"]),
    ("AMP", ["sys_01_amp"]),
    ("AMP\n+compile\n(cold)", ["sys_02_amp_compile_cold", "sys_02_amp_compile"]),
    ("AMP\n+compile\n(warm)", ["sys_03_amp_compile_warm"]),
    ("+async\nI/O", ["sys_04_amp_compile_async", "sys_03_amp_compile_async"]),
    ("+async I/O\nrare ckpt", ["sys_05_async_rare_ckpt"]),
]

PROBE_PATTERN = re.compile(r"^probe_(ts|owt)_bs(\d+)_(fp32|opt)$")

DATASET_LABELS = {"ts": "TinyStories (vocab 10k)", "owt": "OpenWebText (vocab 32k)"}

CARD_VRAM_MB = float(os.environ.get("GPU_VRAM_MB", 11264))

CONTENDED_RUNS = {"owt_recommended", "ts_extended", "owt_extended"}
CONTENDED_PREFIXES = ("owt_wide_",)


def gpu_state(run_name):
    if run_name in CONTENDED_RUNS or run_name.startswith(CONTENDED_PREFIXES):
        return "contended"
    return "idle"


class SkipFigure(Exception):
    pass


class Run:
    def __init__(self, run_dir):
        self.name = run_dir.name
        self.dir = run_dir
        self.config = json.loads((run_dir / "config.json").read_text())
        summary_path = run_dir / "summary.json"
        self.summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        self.rows = []
        with (run_dir / "metrics.jsonl").open() as f:
            for line in f:
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __getitem__(self, key):
        return self.config[key]

    def evals(self):
        steps, losses = [], []
        for row in self.rows:
            v = row.get("val_loss")
            if v is not None and math.isfinite(v):
                steps.append(row["step"])
                losses.append(v)
        return np.array(steps), np.array(losses)

    def curve(self):
        return self.evals()

    def tokens(self, steps):
        return steps * self["batch_size"] * self["context_length"]

    def final_val(self):
        _, losses = self.evals()
        return float(losses[-1]) if len(losses) else float("nan")

    def step_times(self, burn_in=50):
        return np.array([r["step_time"] for r in self.rows if "step_time" in r])[burn_in:]

    def throughput(self):
        return self.summary.get("steady_state_tokens_per_sec", float("nan"))


def load(root, names=None):
    runs = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "metrics.jsonl").exists():
            continue
        if names is not None and run_dir.name not in names:
            continue
        runs[run_dir.name] = Run(run_dir)
    return runs


def pick(runs, *names):
    for name in names:
        if name in runs:
            return runs[name]
    raise SkipFigure(f"none of {list(names)} present in {RUNS_DIR.name}/")


def present_system_stages():
    runs = load(RUNS_DIR)
    stages = []
    for label, candidates in SYSTEM_STAGES:
        for name in candidates:
            if name in runs:
                stages.append((label, runs[name]))
                break
    if not stages:
        raise SkipFigure("no sys_* runs present")
    return stages


def probe_results():
    out = {}
    if not RUNS_DIR.exists():
        return out
    for run_dir in sorted(RUNS_DIR.iterdir()):
        match = PROBE_PATTERN.match(run_dir.name)
        if not match or not run_dir.is_dir():
            continue
        dataset, batch, arm = match.group(1), int(match.group(2)), match.group(3)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            out[(dataset, batch, arm)] = {
                "status": "fits",
                "peak_vram_mb": summary.get("peak_vram_mb"),
                "tokens_per_sec": summary.get("steady_state_tokens_per_sec"),
            }
        else:
            out[(dataset, batch, arm)] = {"status": "oom", "peak_vram_mb": None, "tokens_per_sec": None}
    return out


def wall_time_breakdown(run):
    prev = 0.0
    outside = 0.0
    for row in run.rows:
        outside += row["wall_time"] - prev - row["step_time"]
        prev = row["wall_time"]
    step_times = [row["step_time"] for row in run.rows]
    p50 = float(np.median(run.step_times()))
    spikes = sum(t for t in step_times[1:] if t > 3 * p50)
    total = run.summary.get("total_wall_time", sum(step_times) + outside)
    return {
        "first_step": step_times[0],
        "steady": p50 * (len(step_times) - 1),
        "spikes": spikes,
        "outside": outside,
        "startup": max(total - outside - sum(step_times), 0.0),
        "median_step": p50,
        "total": total,
    }


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def annotate_bars(ax, bars, fmt="{:.4f}"):
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def fig_tinystories_lr():
    runs = sorted(load(RUNS_DIR).values(), key=lambda r: r["learning_rate_base"])
    runs = [r for r in runs if r.name.startswith("ts_lr_")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        steps, losses = run.curve()
        axes[0].plot(steps, losses, label=f"lr={run['learning_rate_base']:.1e}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("TinyStories LR sweep (40.96M tokens)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    lrs = [r["learning_rate_base"] for r in runs]
    finals = [r.final_val() for r in runs]
    stable = [(lr, v) for lr, v in zip(lrs, finals) if v < 3]
    axes[1].plot([lr for lr, _ in stable], [v for _, v in stable], "o-", color="tab:blue")
    low = min(v for _, v in stable)
    high = max(v for _, v in stable)
    top = high + 0.25 * (high - low)
    axes[1].set_ylim(low - 0.1 * (high - low), top)
    for lr, v in zip(lrs, finals):
        if v >= 3:
            axes[1].plot(lr, top, "x", color="tab:red", markersize=10, clip_on=False)
            axes[1].annotate(f"diverged\n({v:.2f})", (lr, top), textcoords="offset points",
                             xytext=(-14, -30), fontsize=8, color="tab:red")
    best = min(stable, key=lambda z: z[1])
    axes[1].annotate(f"best {best[1]:.4f}", best, textcoords="offset points", xytext=(5, 8), fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("Final validation loss")
    axes[1].set_title("Final loss vs learning rate")
    axes[1].grid(alpha=0.3)
    save(fig, "tinystories_lr")


def fig_tinystories_batch_loss():
    runs = sorted(
        (r for r in load(RUNS_DIR).values() if r.name.startswith("ts_batch_")),
        key=lambda r: r["batch_size"],
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        steps, losses = run.curve()
        ax.plot(run.tokens(steps) / 1e6, losses, label=f"bs={run['batch_size']} ({run['num_steps']} steps)")
        ax.annotate(f"{run.final_val():.4f}", (run.tokens(steps)[-1] / 1e6, losses[-1]),
                    textcoords="offset points", xytext=(-38, 4), fontsize=8)
    ax.set_xlabel("Tokens processed (millions)")
    ax.set_ylabel("Validation loss")
    ax.set_title("TinyStories batch size at fixed 40.96M token budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "tinystories_batch_loss")


def fig_tinystories_batch_throughput():
    runs = sorted(
        (r for r in load(RUNS_DIR).values() if r.name.startswith("ts_batch_")),
        key=lambda r: r["batch_size"],
    )
    labels = [str(r["batch_size"]) for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    bars = axes[0].bar(labels, [r.throughput() / 1e3 for r in runs], color="tab:green")
    annotate_bars(axes[0], bars, "{:.0f}k")
    axes[0].set_ylabel("Steady-state tokens/s (thousands)")
    axes[0].set_title("Throughput")

    bars = axes[1].bar(labels, [r.summary.get("total_wall_time", float("nan")) / 60 for r in runs], color="tab:orange")
    annotate_bars(axes[1], bars, "{:.1f}m")
    axes[1].set_ylabel("Wall-clock time (minutes)")
    axes[1].set_title("Time to spend 40.96M tokens")

    bars = axes[2].bar(labels, [r.summary.get("peak_vram_mb", float("nan")) for r in runs], color="tab:purple")
    annotate_bars(axes[2], bars, "{:.0f}")
    axes[2].set_ylabel("Peak VRAM (MB)")
    axes[2].set_title("Memory")

    for ax in axes:
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.3, axis="y")
    save(fig, "tinystories_batch_throughput")


def fig_tinystories_context():
    runs = sorted(
        (r for r in load(RUNS_DIR).values() if r.name.startswith("ts_ctx_")),
        key=lambda r: r["context_length"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        steps, losses = run.curve()
        axes[0].plot(run.tokens(steps) / 1e6, losses,
                     label=f"ctx={run['context_length']} (bs={run['batch_size']})")
    axes[0].set_xlabel("Tokens processed (millions)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Context length at fixed tokens and fixed tokens/update")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    ax2 = axes[1]
    labels = [str(r["context_length"]) for r in runs]
    bars = ax2.bar(labels, [r.final_val() for r in runs], color="tab:blue", alpha=0.85)
    annotate_bars(ax2, bars)
    ax2.set_xlabel("Context length")
    ax2.set_ylabel("Final validation loss")
    ax2.set_ylim(min(r.final_val() for r in runs) - 0.1, max(r.final_val() for r in runs) + 0.05)
    twin = ax2.twinx()
    twin.plot(labels, [r.throughput() / 1e3 for r in runs], "o--", color="tab:red")
    twin.set_ylabel("Steady-state tokens/s (thousands)", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_title("Loss and throughput vs context")
    save(fig, "tinystories_context")


def _final_run_figure(run, title, name, reference=None):
    steps, losses = run.curve()
    tokens = run.tokens(steps) / 1e6
    lrs = [r["lr"] for r in run.rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(tokens, losses, color="tab:blue", label=run.name)
    if reference is not None:
        ref_steps, ref_losses = reference.curve()
        axes[0].plot(reference.tokens(ref_steps) / 1e6, ref_losses, color="tab:gray", alpha=0.7,
                     label=f"{reference.name} ({reference.final_val():.4f})")
        axes[0].legend(fontsize=8)
    axes[0].annotate(f"final {run.final_val():.4f}\nppl {math.exp(run.final_val()):.2f}",
                     (tokens[-1], losses[-1]), textcoords="offset points", xytext=(-70, 12), fontsize=9)
    axes[0].set_xlabel("Tokens processed (millions)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_yscale("log")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(np.arange(1, len(lrs) + 1), lrs, color="tab:orange")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Warmup + cosine schedule")
    axes[1].grid(alpha=0.3)
    save(fig, name)


def fig_tinystories_final():
    runs = load(RUNS_DIR)
    _final_run_figure(runs["ts_final"], "TinyStories full run (327.68M tokens)",
                      "tinystories_final", runs.get("ts_extended"))


def fig_owt_final():
    runs = load(RUNS_DIR)
    primary = pick(runs, "owt_recommended", "owt_final")
    reference = runs.get("owt_final") if primary.name == "owt_recommended" else None
    _final_run_figure(primary, "OpenWebText full run (327.68M tokens)", "owt_final", reference)


def fig_systems_optimizations():
    stages = present_system_stages()
    labels = [label for label, _ in stages]
    runs = [run for _, run in stages]
    parts = [wall_time_breakdown(run) for run in runs]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))

    step_ms = [p["median_step"] * 1e3 for p in parts]
    bars = axes[0].bar(labels, step_ms, color="tab:blue")
    annotate_bars(axes[0], bars, "{:.1f}")
    axes[0].set_ylabel("Median step time (ms)")
    axes[0].set_title(f"Step time ({step_ms[0] / min(step_ms):.2f}x best vs eager FP32)")

    bars = axes[1].bar(labels, [r.throughput() / 1e3 for r in runs], color="tab:green")
    annotate_bars(axes[1], bars, "{:.0f}k")
    axes[1].set_ylabel("Steady-state tokens/s (thousands)")
    axes[1].set_title("Throughput")

    bottom = np.zeros(len(runs))
    for key, label, color in [
        ("steady", "steady-state steps", "tab:blue"),
        ("first_step", "first step (compile / warmup)", "tab:red"),
        ("spikes", "post-checkpoint spikes", "tab:brown"),
        ("outside", "outside timed step (logging, eval)", "tab:orange"),
        ("startup", "startup", "tab:gray"),
    ]:
        values = np.array([p[key] for p in parts])
        axes[2].bar(labels, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[2].set_ylabel("Wall-clock seconds")
    axes[2].set_title("Where the 1000 steps went")
    axes[2].legend(fontsize=7)

    bars = axes[3].bar(labels, [r.summary.get("peak_vram_mb", float("nan")) for r in runs], color="tab:purple")
    annotate_bars(axes[3], bars, "{:.0f}")
    axes[3].set_ylabel("Peak VRAM (MB)")
    axes[3].set_title("Memory")

    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)
    save(fig, "systems_optimizations")


def fig_compile_payback():
    stages = dict(present_system_stages())
    baseline = stages.get("AMP")
    cold = stages.get("AMP\n+compile\n(cold)")
    warm = stages.get("AMP\n+compile\n(warm)")
    if baseline is None or cold is None:
        raise SkipFigure("need the AMP and cold-compile runs")

    steps = np.arange(1, max(len(baseline.rows), len(cold.rows)) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    series = [("AMP, no compile", baseline, "tab:blue"), ("AMP + compile, cold cache", cold, "tab:red")]
    if warm is not None:
        series.append(("AMP + compile, warm cache", warm, "tab:green"))

    curves = {}
    for label, run, color in series:
        part = wall_time_breakdown(run)
        cumulative = part["first_step"] + part["median_step"] * (steps - 1)
        curves[label] = cumulative
        ax.plot(steps, cumulative, color=color, label=f"{label} ({part['median_step'] * 1e3:.1f} ms/step)")

    base = curves["AMP, no compile"]
    for label, color in [("AMP + compile, cold cache", "tab:red"), ("AMP + compile, warm cache", "tab:green")]:
        if label not in curves:
            continue
        crossing = np.argmax(curves[label] < base)
        if crossing > 0:
            ax.axvline(steps[crossing], color=color, linestyle=":", alpha=0.7)
            ax.annotate(f"pays back at\nstep {steps[crossing]}", (steps[crossing], base[crossing]),
                        textcoords="offset points", xytext=(8, -28), fontsize=8, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative wall-clock time (s)")
    ax.set_title("When torch.compile pays for itself")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "compile_payback")


def fig_feasibility():
    results = probe_results()
    if not results:
        raise SkipFigure("no probe_* runs present; run STAGES=feasibility")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dataset in zip(axes, ["ts", "owt"]):
        batches = sorted({b for d, b, _ in results if d == dataset})
        if not batches:
            ax.set_visible(False)
            continue
        width = 0.38
        positions = np.arange(len(batches))
        ceiling = CARD_VRAM_MB
        ax.axhline(CARD_VRAM_MB, color="black", linestyle="--", linewidth=1)
        ax.annotate(f"{CARD_VRAM_MB:,.0f} MiB card", (0, CARD_VRAM_MB), textcoords="offset points",
                    xytext=(2, -12), fontsize=8)

        for offset, arm, color in [(-width / 2, "fp32", "tab:red"), (width / 2, "opt", "tab:green")]:
            heights, hatches = [], []
            for batch in batches:
                entry = results.get((dataset, batch, arm))
                if entry is None:
                    heights.append(0.0)
                    hatches.append("")
                elif entry["status"] == "oom":
                    heights.append(ceiling)
                    hatches.append("//")
                else:
                    heights.append(entry["peak_vram_mb"] or 0.0)
                    hatches.append("")
            bars = ax.bar(positions + offset, heights, width,
                          label={"fp32": "eager FP32", "opt": "AMP + compile + async"}[arm], color=color)
            for bar, hatch, batch in zip(bars, hatches, batches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.45)
                    ax.annotate("OOM", (bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5),
                                ha="center", fontsize=9, weight="bold", color="darkred")
                else:
                    ax.annotate(f"{bar.get_height():.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha="center", va="bottom", fontsize=8)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"batch {b}" for b in batches])
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Does the recommended configuration fit? (200-step probes)")
    save(fig, "feasibility")


def fig_owt_lr_wide():
    runs = load(RUNS_DIR)
    wide = sorted((r for r in runs.values() if r.name.startswith("owt_wide_")),
                  key=lambda r: r["learning_rate_base"])
    if not wide:
        raise SkipFigure("no owt_wide_* runs present; run STAGES=owt_lr_wide")
    transfer = sorted((r for r in runs.values() if r.name.startswith("owt_lr_")),
                      key=lambda r: r["learning_rate_base"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in wide:
        steps, losses = run.curve()
        axes[0].plot(run.tokens(steps) / 1e6, losses, label=f"lr={run['learning_rate_base']:.2e}")
    axes[0].set_xlabel("Tokens processed (millions)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title(f"OpenWebText LR bracket at batch {wide[0]['batch_size']} (40.96M tokens)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot([r["learning_rate_base"] for r in wide], [r.final_val() for r in wide],
                 "o-", color="tab:blue", label=f"batch {wide[0]['batch_size']} (wide bracket)")
    if transfer:
        axes[1].plot([r["learning_rate_base"] for r in transfer], [r.final_val() for r in transfer],
                     "s--", color="tab:gray", alpha=0.8,
                     label=f"batch {transfer[0]['batch_size']} (transfer test)")
    best = min(wide, key=lambda r: r.final_val())
    axes[1].annotate(f"best {best.final_val():.4f}\nlr={best['learning_rate_base']:.2e}",
                     (best["learning_rate_base"], best.final_val()),
                     textcoords="offset points", xytext=(6, 10), fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("Final validation loss")
    axes[1].set_title("Optimum depends on batch size and budget")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    save(fig, "owt_lr_wide")


def fig_extended_budget():
    runs = load(RUNS_DIR)
    panels = [
        ("TinyStories", ["ts_final"], "ts_extended"),
        ("OpenWebText", ["owt_recommended", "owt_final"], "owt_extended"),
    ]
    available = [(t, b, e) for t, b, e in panels if e in runs]
    if not available:
        raise SkipFigure("no *_extended runs present; run STAGES='ts_extended owt_extended'")

    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 4.5), squeeze=False)
    for ax, (title, baseline_names, extended_name) in zip(axes[0], available):
        for name, color, style in [(extended_name, "tab:blue", "-")]:
            run = runs[name]
            steps, losses = run.curve()
            ax.plot(run.tokens(steps) / 1e6, losses, style, color=color,
                    label=f"{name} ({run.final_val():.4f})")
        for name in baseline_names:
            if name not in runs:
                continue
            run = runs[name]
            steps, losses = run.curve()
            ax.plot(run.tokens(steps) / 1e6, losses, "--", color="tab:gray", alpha=0.85,
                    label=f"{name} ({run.final_val():.4f})")
            break
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tokens processed (millions)")
        ax.set_ylabel("Validation loss")
        ax.set_title(f"{title}: doubled token budget")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    save(fig, "extended_budget")


def fig_owt_lr_transfer():
    runs = load(RUNS_DIR)
    labels = ["0.5x", "1.0x", "2.0x"]
    names = ["owt_lr_half", "owt_lr_same", "owt_lr_double"]
    selected = [runs[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, run in zip(labels, selected):
        steps, losses = run.curve()
        axes[0].plot(run.tokens(steps) / 1e6, losses,
                     label=f"{label} TinyStories LR (lr={run['learning_rate_base']:.2e})")
    axes[0].set_xlabel("Tokens processed (millions)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("OpenWebText LR transfer (40.96M tokens)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    finals = [r.final_val() for r in selected]
    bars = axes[1].bar(labels, finals, color=["tab:blue", "tab:green", "tab:red"])
    annotate_bars(axes[1], bars)
    axes[1].set_ylim(min(finals) - 0.05, max(finals) + 0.05)
    axes[1].set_xlabel("Learning rate, relative to TinyStories optimum")
    axes[1].set_ylabel("Final validation loss")
    axes[1].set_title("Transfer penalty is asymmetric")
    axes[1].grid(alpha=0.3, axis="y")
    save(fig, "owt_lr_transfer")


def fig_eval_noise_floor():
    replicates = ["ts_lr_1p5e_3", "ts_batch_32", "ts_ctx_256"]
    runs = load(RUNS_DIR, set(replicates))
    missing = [n for n in replicates if n not in runs]
    if missing:
        raise SkipFigure(f"missing replicate runs: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in replicates:
        steps, losses = runs[name].evals()
        axes[0].plot(steps, losses, alpha=0.85, label=f"{name} ({losses[-1]:.4f})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Validation loss (20 fixed batches)")
    axes[0].set_title("Three identical configurations, full curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    tail_values = []
    for name in replicates:
        steps, losses = runs[name].evals()
        keep = steps >= 0.6 * steps[-1]
        axes[1].plot(steps[keep], losses[keep], marker="o", markersize=3, alpha=0.85, label=name)
        tail_values.extend(losses[keep].tolist())
    finals = [runs[n].final_val() for n in replicates]
    axes[1].set_ylim(min(tail_values) - 0.005, max(tail_values) + 0.005)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title(f"Final spread = {max(finals) - min(finals):.4f} (last 40%, zoomed)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    save(fig, "eval_noise_floor")


def token_stats():
    path = PROJECT_ROOT / "results" / "token_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


def bits_per_byte(run, stats):
    key = "openwebtext" if "owt" in run["train_path_text"] else "tinystories"
    entry = stats.get(key, {}).get(str(run["vocab_size"]))
    if not entry:
        return None
    return run.final_val() / math.log(2) * entry["tokens"] / entry["bytes"]


def tables():
    runs = load(RUNS_DIR)
    stats = token_stats()

    def row(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"

    probes = probe_results()
    if probes:
        print("\n### 4 feasibility probes (200 steps)\n")
        print(row(["Dataset", "Vocabulary", "Batch", "Precision / stack", "Outcome", "Peak VRAM", "Tokens/s"]))
        print(row(["---", "---:", "---:", "---", "---", "---:", "---:"]))
        for (dataset, batch, arm) in sorted(probes):
            entry = probes[(dataset, batch, arm)]
            vram = f"{entry['peak_vram_mb']:.0f} MB" if entry["peak_vram_mb"] else "n/a"
            tps = f"{entry['tokens_per_sec']:,.0f}" if entry["tokens_per_sec"] else "n/a"
            print(row([DATASET_LABELS[dataset].split(" (")[0], 10000 if dataset == "ts" else 32000, batch,
                       "eager FP32" if arm == "fp32" else "AMP + compile + async",
                       "fits" if entry["status"] == "fits" else "**OOM**", vram, tps]))

    print("\n### 4.1 learning rate\n")
    print(row(["Run", "Learning rate", "Final validation loss", "Status"]))
    print(row(["---", "---:", "---:", "---"]))
    for run in sorted((r for r in runs.values() if r.name.startswith("ts_lr_")),
                      key=lambda r: r["learning_rate_base"]):
        v = run.final_val()
        status = "diverged" if v > 3 else "stable"
        print(row([f"`{run.name}`", f"{run['learning_rate_base']:.1e}", f"{v:.4f}", status]))

    print("\n### 4.2 batch size\n")
    print(row(["Batch size", "Steps", "Tokens processed", "Final validation loss", "Tokens/s", "Wall clock", "Peak VRAM"]))
    print(row(["---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for run in sorted((r for r in runs.values() if r.name.startswith("ts_batch_")),
                      key=lambda r: r["batch_size"]):
        print(row([run["batch_size"], run["num_steps"], f"{run.summary['total_tokens'] / 1e6:.2f}M",
                   f"{run.final_val():.4f}", f"{run.throughput():,.0f}",
                   f"{run.summary['total_wall_time'] / 60:.1f} min",
                   f"{run.summary['peak_vram_mb']:.0f} MB"]))

    print("\n### 4.3 context length\n")
    print(row(["Context", "Batch size", "Steps", "Tokens/update", "Final validation loss", "Tokens/s", "Peak VRAM"]))
    print(row(["---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for run in sorted((r for r in runs.values() if r.name.startswith("ts_ctx_")),
                      key=lambda r: r["context_length"]):
        print(row([run["context_length"], run["batch_size"], run["num_steps"],
                   run["batch_size"] * run["context_length"], f"{run.final_val():.4f}",
                   f"{run.throughput():,.0f}", f"{run.summary['peak_vram_mb']:.0f} MB"]))

    print("\n### 4.2 systems stack\n")
    print(row(["Stage", "Run", "Median step", "First step", "Post-ckpt spikes", "Outside step",
               "End-to-end", "Tokens/s", "Peak VRAM", "Final val loss"]))
    print(row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    try:
        stages = present_system_stages()
    except SkipFigure:
        stages = []
    for label, run in stages:
        parts = wall_time_breakdown(run)
        print(row([label.replace("\n", " "), f"`{run.name}`", f"{parts['median_step'] * 1e3:.1f} ms",
                   f"{parts['first_step']:.1f} s", f"{parts['spikes']:.1f} s", f"{parts['outside']:.1f} s",
                   f"{parts['total']:.1f} s", f"{run.throughput():,.0f}",
                   f"{run.summary['peak_vram_mb']:.0f} MB", f"{run.final_val():.4f}"]))

    wide = sorted((r for r in runs.values() if r.name.startswith("owt_wide_")),
                  key=lambda r: r["learning_rate_base"])
    if wide:
        print("\n### 6.3 OWT learning-rate bracket (batch 32, 40.96M tokens)\n")
        print(row(["Run", "Learning rate", "Final validation loss", "Status"]))
        print(row(["---", "---:", "---:", "---"]))
        for run in wide:
            v = run.final_val()
            print(row([f"`{run.name}`", f"{run['learning_rate_base']:.2e}", f"{v:.4f}",
                       "diverged" if v > 6 else "stable"]))

    print("\n### 6.1 OWT learning-rate transfer\n")
    print(row(["Run", "Learning rate", "Final validation loss", "Delta vs best"]))
    print(row(["---", "---:", "---:", "---:"]))
    transfer = [("0.5x TinyStories LR", "owt_lr_half"), ("1.0x TinyStories LR", "owt_lr_same"),
                ("2.0x TinyStories LR", "owt_lr_double")]
    best = min(runs[n].final_val() for _, n in transfer)
    for label, name in transfer:
        run = runs[name]
        print(row([label, f"{run['learning_rate_base']:.2e}", f"{run.final_val():.4f}",
                   f"{run.final_val() - best:+.4f}"]))

    print("\n### 5.4 / 6.2 / 7 / 8 final and extended runs\n")
    print(row(["Dataset", "Run", "Vocabulary", "Batch", "Steps", "Tokens", "Val loss", "Perplexity",
               "Bits/byte", "Peak VRAM", "Wall clock", "Tokens/s", "GPU"]))
    print(row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    for label, name in [("TinyStories", "ts_final"), ("TinyStories", "ts_extended"),
                        ("OpenWebText", "owt_final"), ("OpenWebText", "owt_recommended"),
                        ("OpenWebText", "owt_extended")]:
        run = runs.get(name)
        if run is None:
            continue
        v = run.final_val()
        bpb = bits_per_byte(run, stats)
        print(row([label, f"`{run.name}`", run["vocab_size"], run["batch_size"], run["num_steps"],
                   f"{run.summary['total_tokens'] / 1e6:.2f}M", f"{v:.4f}", f"{math.exp(v):.2f}",
                   f"{bpb:.3f}" if bpb else "n/a", f"{run.summary['peak_vram_mb']:.0f} MB",
                   f"{run.summary['total_wall_time'] / 60:.1f} min", f"{run.throughput():,.0f}",
                   gpu_state(run.name)]))
    if not stats:
        print("\n<!-- bits/byte needs results/token_stats.json:")
        print('     {"openwebtext": {"32000": {"tokens": <bin_bytes/2>, "bytes": <txt_bytes>}}, ...} -->')


FIGURES = {
    "feasibility": fig_feasibility,
    "compile_payback": fig_compile_payback,
    "owt_lr_wide": fig_owt_lr_wide,
    "extended_budget": fig_extended_budget,
    "tinystories_lr": fig_tinystories_lr,
    "tinystories_batch_loss": fig_tinystories_batch_loss,
    "tinystories_batch_throughput": fig_tinystories_batch_throughput,
    "tinystories_context": fig_tinystories_context,
    "tinystories_final": fig_tinystories_final,
    "systems_optimizations": fig_systems_optimizations,
    "owt_lr_transfer": fig_owt_lr_transfer,
    "owt_final": fig_owt_final,
    "eval_noise_floor": fig_eval_noise_floor,
}


def main(argv):
    if "--list" in argv:
        print("\n".join(FIGURES))
        return 0
    if "--tables" in argv:
        tables()
        return 0
    wanted = [a for a in argv if not a.startswith("-")] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        print(f"unknown figures: {unknown}\navailable: {list(FIGURES)}")
        return 1
    for name in wanted:
        try:
            FIGURES[name]()
        except SkipFigure as reason:
            print(f"skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
