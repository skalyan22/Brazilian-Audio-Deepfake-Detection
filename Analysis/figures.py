import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCORES_DIR = "results/scores"
FIGS_DIR = "paper/figs"
MIN_SPEAKERS = 10

MODEL_LABEL = {"cnn": "Mel-CNN", "w2v": "Wav2Vec2"}


class Palette:
    BLUE_FILL = "#aec7e8"
    RED_FILL = "#f4a09a"
    BLUE_LINE = "#3a6ea5"
    RED_LINE = "#c44e52"


class FigureStyle:
    RC_PARAMS = {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": "--",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.edgecolor": "#444444",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
    }
    BOX = dict(linewidth=0.9, color="black")
    MEAN = dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                markersize=4.5)
    MEDIAN = dict(color="black", linewidth=1.1)

    @classmethod
    def apply(cls):
        plt.rcParams.update(cls.RC_PARAMS)

    @classmethod
    def boxplot(cls, ax, data, positions, width, color):
        return ax.boxplot(data, positions=positions, widths=width,
                          patch_artist=True, showmeans=True, showfliers=False,
                          boxprops=dict(facecolor=color, **cls.BOX),
                          whiskerprops=cls.BOX, capprops=cls.BOX,
                          meanprops=cls.MEAN, medianprops=cls.MEDIAN)

    @staticmethod
    def legend_above(ax, *args, **kwargs):
        ax.legend(*args, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, fontsize=7, columnspacing=0.9, **kwargs)


def save(fig, name):
    fig.savefig(f"{FIGS_DIR}/{name}.pdf")
    fig.savefig(f"{FIGS_DIR}/{name}.png", dpi=300)
    print(f"wrote {name}.pdf + {name}.png")


def plot_reliability():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=0.8, label="Perfect calibration")
    for model, color in (("cnn", Palette.BLUE_LINE), ("w2v", Palette.RED_LINE)):
        scores_df = pd.read_csv(f"{SCORES_DIR}/{model}_scores_test.csv")
        scores = scores_df["score"].values
        labels = scores_df["label"].values
        confidence = np.maximum(scores, 1 - scores)
        correct = ((scores > 0.5).astype(int) == labels).astype(float)
        bins = np.linspace(0.5, 1.0, 11)
        xs, accuracies = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidence >= lo) & (confidence <= hi if hi == 1.0 else confidence < hi)
            if mask.sum() >= 5:
                xs.append(confidence[mask].mean())
                accuracies.append(correct[mask].mean())
        ax.plot(xs, accuracies, "o-", ms=4, lw=1.3, color=color,
                markerfacecolor="white", markeredgecolor=color,
                label=MODEL_LABEL[model])
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Empirical accuracy")
    FigureStyle.legend_above(ax)
    fig.tight_layout()
    save(fig, "fig_reliability")


def plot_score_shift():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    centers = [0, 1]
    width, offset = 0.32, 0.20
    box_indomain, box_ood = None, None
    for i, model in enumerate(("cnn", "w2v")):
        test = pd.read_csv(f"{SCORES_DIR}/{model}_scores_test.csv")
        ood = pd.read_csv(f"{SCORES_DIR}/{model}_scores_brspeech.csv")
        indomain_scores = test[test.label == 1]["score"].values
        ood_scores = ood[ood.label_int == 1]["score"].values
        box_indomain = FigureStyle.boxplot(
            ax, [indomain_scores], [centers[i] - offset], width, Palette.BLUE_FILL)
        box_ood = FigureStyle.boxplot(
            ax, [ood_scores], [centers[i] + offset], width, Palette.RED_FILL)
        for values, position in ((indomain_scores, centers[i] - offset),
                                 (ood_scores, centers[i] + offset)):
            q3 = np.percentile(values, 75)
            q1 = np.percentile(values, 25)
            whisker_top = min(values.max(), q3 + 1.5 * (q3 - q1))
            ax.annotate(f"{np.median(values):.2f}", (position, whisker_top),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABEL["cnn"], MODEL_LABEL["w2v"]])
    ax.set_ylabel("Spoof score $P(\\mathrm{fake})$")
    ax.set_ylim(-0.05, 1.1)
    FigureStyle.legend_above(
        ax, [box_indomain["boxes"][0], box_ood["boxes"][0]],
        ["In-domain spoof (test)", "OOD spoof (BRSpeech-DF)"])
    fig.tight_layout()
    save(fig, "fig_score_shift")


def plot_region_fpr():
    with open(f"{SCORES_DIR}/cnn_results_fairness.json") as f:
        by_region = json.load(f)["mupe_by_region"]
    regions = [r for r in ("North", "Northeast", "Central-West",
                           "Southeast", "South")
               if by_region[r]["n_speakers"] >= MIN_SPEAKERS]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    width = 0.34
    x = np.arange(len(regions))
    speakers_by_region = {}
    for i, (model, fill) in enumerate((("cnn", Palette.BLUE_FILL),
                                       ("w2v", Palette.RED_FILL))):
        with open(f"{SCORES_DIR}/{model}_results_fairness.json") as f:
            region_stats = json.load(f)["mupe_by_region"]
        speakers_by_region = {r: region_stats[r]["n_speakers"] for r in regions}
        values = [100 * region_stats[r]["fpr"] for r in regions]
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=fill,
                      edgecolor="black", linewidth=0.8, label=MODEL_LABEL[model])
        for rect, value in zip(bars, values):
            ax.annotate(f"{value:.1f}",
                        (rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r}\n({speakers_by_region[r]} speakers)" for r in regions],
        fontsize=7.5)
    ax.set_ylabel("Bona fide FPR (%)")
    ax.set_ylim(0, 14.5)
    FigureStyle.legend_above(ax)
    fig.tight_layout()
    save(fig, "fig_region_fpr")


def plot_scaling():
    sizes = [2493, 6233, 12466, 24931]
    test_eer = [1.50, 1.29, 0.58, 0.19]
    brspeech_eer = [41.0, 41.7, 36.5, 35.2]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for values, color, label in ((test_eer, Palette.BLUE_LINE, "In-domain test"),
                                 (brspeech_eer, Palette.RED_LINE,
                                  "OOD (BRSpeech-DF)")):
        ax.plot(sizes, values, "o-", lw=1.4, ms=5, color=color,
                markerfacecolor="white", markeredgecolor=color, label=label)
        for size, value in zip(sizes, values):
            ax.annotate(f"{value:.1f}%", (size, value), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=color)
    ax.set_xscale("log")
    ax.set_xlim(2000, 32000)
    ax.set_xticks(sizes)
    ax.set_xticklabels(["2.5K", "6.2K", "12.5K", "25K"], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Training clips (all 4 engines, log scale)")
    ax.set_ylabel("EER (%)")
    ax.set_ylim(-4, 55)
    FigureStyle.legend_above(ax)
    fig.tight_layout()
    save(fig, "fig_scaling")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    FigureStyle.apply()
    plot_reliability()
    plot_score_shift()
    plot_region_fpr()
    plot_scaling()


if __name__ == "__main__":
    main()
