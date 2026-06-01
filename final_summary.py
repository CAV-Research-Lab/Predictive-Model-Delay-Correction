"""
Final summary figure: final-episode tracking error vs delay.

Metric = true euclidean tracking error (REnvPDNormObs.reward) over the last 100 logged
training episodes. This is the SAME metric the thesis reports in Table 2 ("mean and
standard deviation over the final training episode").

NOTE: we deliberately do NOT use a reloaded-policy deterministic rollout for PMDC, because
the PMDC ensemble world model is trained online and is NOT saved inside the SAC .zip.
Reloading the policy into a fresh PMDC wrapper gives it a random dynamics model, which makes
PMDC fail catastrophically (~1.9 m). A-SAC/SAC have no world model so they reload fine, but
for a like-for-like comparison we read all three from the live training logs.

Numbers are read from /tmp/real_results.json produced by the TensorBoard analysis, so this
script never hardcodes results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("/tmp/real_results.json") as f:
    results = json.load(f)  # {delay: {alg: [mean, std]}} in metres (mean is negative reward)

# Thesis Table 2 reference values (final training episode), metres.
thesis = {
    "90-120ms":  {"PMDC": 0.030, "A-SAC": 0.034, "SAC": 0.053},
    "250-290ms": {"PMDC": 0.043, "A-SAC": 0.150, "SAC": 0.250},
}

COLORS = {"PMDC": "#1f77b4", "A-SAC": "#ff7f0e", "SAC": "#2ca02c"}
algorithms = ["PMDC", "A-SAC", "SAC"]
delays = ["90-120ms", "250-290ms"]


def err_cm(delay, alg):
    m, s = results[delay][alg]
    return abs(m) * 100, s * 100


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Left: grouped bar chart (our reproduction) ---
ax = axes[0]
x = np.arange(len(delays))
width = 0.25
for i, alg in enumerate(algorithms):
    means = [err_cm(d, alg)[0] for d in delays]
    stds = [err_cm(d, alg)[1] for d in delays]
    bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds,
                  label=alg + (" (ours)" if alg == "PMDC" else ""),
                  color=COLORS[alg], capsize=4, alpha=0.9)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(delays, fontsize=12)
ax.set_ylabel("Final-episode tracking error (cm)\n[lower = better]", fontsize=12)
ax.set_title("Reproduction: final-episode tracking error\n(FetchPush local-remote, 60k steps, seed 0)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.annotate("all within ~1.5 cm\n(delay manageable)", xy=(0, 6), xytext=(0.05, 14),
            ha="center", fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6))
asac_250 = err_cm("250-290ms", "A-SAC")[0]
pmdc_250 = err_cm("250-290ms", "PMDC")[0]
ax.annotate(f"PMDC {asac_250 / pmdc_250:.1f}x better\n+ far more stable", xy=(1, asac_250),
            xytext=(0.7, asac_250 + 10), ha="center", fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.8))

# --- Right: scaling line plot (error vs delay), ours solid vs thesis dashed ---
ax = axes[1]
delay_x = [105, 270]
for alg in algorithms:
    y = [err_cm(d, alg)[0] for d in delays]
    yerr = [err_cm(d, alg)[1] for d in delays]
    ax.errorbar(delay_x, y, yerr=yerr, marker="o", markersize=9, linewidth=2.5,
                label=alg + (" (ours)" if alg == "PMDC" else ""),
                color=COLORS[alg], capsize=5)
    yt = [thesis[d][alg] * 100 for d in delays]
    ax.plot(delay_x, yt, marker="s", markersize=6, linewidth=1.2, linestyle="--",
            color=COLORS[alg], alpha=0.45)

ax.set_xlabel("Total delay (ms)", fontsize=12)
ax.set_ylabel("Final-episode tracking error (cm)", fontsize=12)
ax.set_title("PMDC's advantage grows with delay length\n(solid = our runs, dashed = thesis Table 2)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(60, 310)

plt.tight_layout()
plt.savefig("PMDC_reproduction_summary.png", dpi=150, bbox_inches="tight")
print("Saved: PMDC_reproduction_summary.png")

print("\n" + "=" * 70)
print("FINAL REPRODUCTION RESULTS (final-episode true tracking error, seed 0)")
print("=" * 70)
print(f"{'Delay':<12}{'Algorithm':<8}{'Ours mean+/-std (cm)':<24}{'Thesis (cm)':<12}{'x vs PMDC'}")
print("-" * 70)
for d in delays:
    pmdc_mean = err_cm(d, "PMDC")[0]
    for alg in algorithms:
        m, s = err_cm(d, alg)
        print(f"{d:<12}{alg:<8}{m:5.1f} +/- {s:4.1f}{'':<11}{thesis[d][alg]*100:5.1f}{'':<7}{m/pmdc_mean:.2f}x")
    print()
