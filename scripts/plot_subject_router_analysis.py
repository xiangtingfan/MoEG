import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_accuracy(summary_path, num_subjects):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    acc = np.zeros(num_subjects, dtype=np.float64)
    for subject, result in summary["results"].items():
        acc[int(subject)] = float(result["best_target_acc"])
    return acc, float(summary["mean_acc"])


def load_router(run_dir, target):
    target_dir = os.path.join(run_dir, f"target_{target:02d}")
    weights = np.load(os.path.join(target_dir, "best_router_weights.npy"))
    if weights.ndim == 3:
        weights = weights.mean(axis=1)
    top1 = weights.argmax(axis=-1)
    return weights, top1


def entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = prob / np.maximum(prob.sum(axis=-1, keepdims=True), 1e-12)
    return -np.sum(prob * np.log(prob + 1e-12), axis=-1)


def collect_subject_stats(args):
    avg_weights = []
    top1_rates = []
    usage_entropy = []
    for target in range(args.num_subjects):
        weights, top1 = load_router(args.run_dir, target)
        avg = weights.mean(axis=0)
        top1_rate = np.bincount(top1, minlength=args.num_experts).astype(np.float64) / len(top1)
        avg_weights.append(avg)
        top1_rates.append(top1_rate)
        usage_entropy.append(float(entropy(avg)))
    return np.asarray(avg_weights), np.asarray(top1_rates), np.asarray(usage_entropy)


def style_axes(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_per_subject_bars(matrix, acc, output_dir, prefix, ylabel, title):
    outputs = []
    colors = ["#7F8F84", "#B7A99A", "#8A9199", "#B88C8C", "#9CA986", "#C4A69D"]
    for subject in range(matrix.shape[0]):
        fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=220)
        bars = ax.bar(
            [f"E{i}" for i in range(matrix.shape[1])],
            matrix[subject],
            color=colors[: matrix.shape[1]],
            width=0.62,
        )
        ax.set_ylim(0, max(1.0, float(matrix.max()) * 1.12))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} S{subject:02d} | acc {acc[subject] * 100:.2f}%")
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in matrix[subject]], padding=3, fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        out = os.path.join(output_dir, f"{prefix}_subject_{subject:02d}.png")
        fig.savefig(out)
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_heatmap_with_accuracy(avg_weights, acc, mean_acc, output_dir):
    matrix = avg_weights
    fig = plt.figure(figsize=(8.4, 7.2), dpi=220)
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 0.35, 1.4], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    acc_ax = fig.add_subplot(gs[0, 2])

    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1e-8, float(matrix.max())))
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f"E{i}" for i in range(matrix.shape[1])])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"S{i:02d}" for i in range(matrix.shape[0])])
    ax.set_xlabel("Expert")
    ax.set_ylabel("Target subject")
    ax.set_title("Subject-expert mean router weight")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#222222")
    fig.colorbar(image, cax=cax)

    y = np.arange(len(acc))
    acc_ax.barh(y, acc * 100, color="#B88C8C", height=0.72)
    acc_ax.axvline(mean_acc * 100, color="#555555", linestyle="--", linewidth=1.2)
    acc_ax.set_ylim(ax.get_ylim())
    acc_ax.set_yticks([])
    acc_ax.set_xlabel("Acc (%)", fontsize=8)
    acc_ax.set_xlim(0, 100)
    acc_ax.tick_params(axis="x", labelsize=7)
    acc_ax.spines["top"].set_visible(False)
    acc_ax.spines["right"].set_visible(False)
    acc_ax.spines["left"].set_visible(False)
    acc_ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)

    out = os.path.join(output_dir, "subject_expert_heatmap_with_accuracy.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_entropy_accuracy(entropy_values, acc, output_dir):
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=220)
    ax.scatter(entropy_values, acc * 100, color="#7F8F84", s=54, alpha=0.86)
    for subject, (x, y) in enumerate(zip(entropy_values, acc * 100)):
        ax.text(x + 0.005, y + 0.45, f"S{subject:02d}", fontsize=7)
    coef = np.polyfit(entropy_values, acc * 100, deg=1)
    xs = np.linspace(float(entropy_values.min()), float(entropy_values.max()), 100)
    ax.plot(xs, coef[0] * xs + coef[1], color="#B88C8C", linestyle="--", linewidth=1.4)
    corr = np.corrcoef(entropy_values, acc * 100)[0, 1]
    ax.set_xlabel("Expert usage entropy")
    ax.set_ylabel("Target accuracy (%)")
    ax.set_title(f"Router entropy vs accuracy (r={corr:.2f})")
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(output_dir, "expert_usage_entropy_vs_accuracy.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_subject_cluster(avg_weights, acc, output_dir):
    distances = pdist(avg_weights, metric="euclidean")
    z = linkage(distances, method="ward")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=220)
    labels = [f"S{i:02d}\n{acc[i] * 100:.1f}%" for i in range(avg_weights.shape[0])]
    dendrogram(z, labels=labels, ax=ax, leaf_rotation=0, color_threshold=None)
    ax.set_title("Subject clustering by router expert distribution")
    ax.set_ylabel("Ward distance")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(output_dir, "subject_router_distribution_cluster.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_overview_bar_grid(paths, output_path):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w = 360
    thumbs = []
    for path, image in zip(paths, images):
        scale = thumb_w / image.width
        thumbs.append((path, image.resize((thumb_w, int(image.height * scale)))))

    cols = 3
    rows = int(np.ceil(len(thumbs) / cols))
    pad = 14
    label_h = 24
    width = cols * thumb_w + (cols + 1) * pad
    height = rows * (thumbs[0][1].height + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (path, image) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * (image.height + label_h + pad)
        draw.text((x, y), os.path.basename(path).replace(".png", ""), fill=(40, 40, 40))
        canvas.paste(image, (x, y + label_h))
    canvas.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="results/seediv_sample_style_sparse_moge_loso")
    parser.add_argument("--summary_path", default="results/seediv_sample_style_sparse_moge_loso/summary.json")
    parser.add_argument("--output_dir", default="results/seediv_router_subject_analysis")
    parser.add_argument("--num_subjects", type=int, default=15)
    parser.add_argument("--num_experts", type=int, default=4)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    acc, mean_acc = load_accuracy(args.summary_path, args.num_subjects)
    avg_weights, top1_rates, entropy_values = collect_subject_stats(args)

    avg_paths = plot_per_subject_bars(
        avg_weights,
        acc,
        output_dir,
        "mean_router_weight",
        "Mean router weight",
        "Mean expert weight",
    )
    top1_paths = plot_per_subject_bars(
        top1_rates,
        acc,
        output_dir,
        "top1_expert_rate",
        "Top-1 expert rate",
        "Top-1 expert usage",
    )
    outputs = []
    outputs.extend(avg_paths)
    outputs.extend(top1_paths)
    outputs.append(save_overview_bar_grid(avg_paths, os.path.join(output_dir, "overview_mean_router_weight_by_subject.png")))
    outputs.append(save_overview_bar_grid(top1_paths, os.path.join(output_dir, "overview_top1_expert_rate_by_subject.png")))
    outputs.append(plot_heatmap_with_accuracy(avg_weights, acc, mean_acc, output_dir))
    outputs.append(plot_entropy_accuracy(entropy_values, acc, output_dir))
    outputs.append(plot_subject_cluster(avg_weights, acc, output_dir))

    np.savez(
        os.path.join(output_dir, "subject_router_stats.npz"),
        avg_weights=avg_weights,
        top1_rates=top1_rates,
        entropy=entropy_values,
        accuracy=acc,
        mean_accuracy=mean_acc,
    )
    print("\n".join(path for path in outputs if path))


if __name__ == "__main__":
    main()
