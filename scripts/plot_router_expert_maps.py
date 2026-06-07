import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_target(run_dir, target):
    target_dir = os.path.join(run_dir, f"target_{target:02d}")
    weights = np.load(os.path.join(target_dir, "best_router_weights.npy"))
    labels = np.load(os.path.join(target_dir, "best_router_labels.npy"))
    subject_ids = np.load(os.path.join(target_dir, "best_router_subject_ids.npy"))
    if weights.ndim == 3:
        weights = weights.mean(axis=1)
    return weights, labels, subject_ids


def plot_heatmap(matrix, row_labels, col_labels, title, output_path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=220)
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1e-8, float(matrix.max())))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8, color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def per_subject_label_maps(args, output_dir):
    outputs = []
    all_avg_rows = []
    all_top1_rows = []
    for target in range(args.num_subjects):
        weights, labels, _ = load_target(args.run_dir, target)
        experts = weights.shape[-1]
        avg = np.zeros((args.num_classes, experts), dtype=np.float64)
        top1_rate = np.zeros((args.num_classes, experts), dtype=np.float64)
        top1 = weights.argmax(axis=-1)
        for label in range(args.num_classes):
            mask = labels == label
            if mask.any():
                avg[label] = weights[mask].mean(axis=0)
                top1_rate[label] = np.bincount(top1[mask], minlength=experts) / mask.sum()
        all_avg_rows.append(avg)
        all_top1_rows.append(top1_rate)

        out_avg = os.path.join(output_dir, f"target_{target:02d}_label_expert_avg_weight.png")
        plot_heatmap(
            avg,
            [f"class {i}" for i in range(args.num_classes)],
            [f"E{i}" for i in range(experts)],
            f"Target subject {target:02d}: mean router weight by label",
            out_avg,
        )
        outputs.append(out_avg)

        out_top1 = os.path.join(output_dir, f"target_{target:02d}_label_expert_top1_rate.png")
        plot_heatmap(
            top1_rate,
            [f"class {i}" for i in range(args.num_classes)],
            [f"E{i}" for i in range(experts)],
            f"Target subject {target:02d}: top-1 expert rate by label",
            out_top1,
        )
        outputs.append(out_top1)
    return outputs, np.asarray(all_avg_rows), np.asarray(all_top1_rows)


def plot_subject_expert_summary(args, output_dir):
    subject_avg = []
    subject_top1 = []
    label_avg_sum = np.zeros((args.num_classes, args.num_experts), dtype=np.float64)
    label_top1_sum = np.zeros((args.num_classes, args.num_experts), dtype=np.float64)
    label_counts = np.zeros(args.num_classes, dtype=np.float64)

    for target in range(args.num_subjects):
        weights, labels, _ = load_target(args.run_dir, target)
        if weights.shape[-1] != args.num_experts:
            args.num_experts = weights.shape[-1]
        subject_avg.append(weights.mean(axis=0))
        top1 = weights.argmax(axis=-1)
        subject_top1.append(np.bincount(top1, minlength=weights.shape[-1]) / len(top1))
        for label in range(args.num_classes):
            mask = labels == label
            if mask.any():
                label_avg_sum[label] += weights[mask].sum(axis=0)
                label_top1_sum[label] += np.eye(weights.shape[-1])[top1[mask]].sum(axis=0)
                label_counts[label] += mask.sum()

    subject_avg = np.asarray(subject_avg)
    subject_top1 = np.asarray(subject_top1)
    label_avg = label_avg_sum / np.maximum(label_counts[:, None], 1)
    label_top1 = label_top1_sum / np.maximum(label_counts[:, None], 1)

    outputs = []
    for name, matrix, ylabel, fmt in [
        ("all_subjects_expert_avg_weight", subject_avg, "Target subject", ".2f"),
        ("all_subjects_expert_top1_rate", subject_top1, "Target subject", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=220)
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1e-8, float(matrix.max())))
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels([f"E{i}" for i in range(matrix.shape[1])])
        ax.set_yticklabels([f"S{i:02d}" for i in range(matrix.shape[0])])
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Expert")
        ax.set_ylabel(ylabel)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = os.path.join(output_dir, f"{name}.png")
        fig.savefig(out)
        plt.close(fig)
        outputs.append(out)

    outputs.append(os.path.join(output_dir, "all_labels_expert_avg_weight.png"))
    plot_heatmap(
        label_avg,
        [f"class {i}" for i in range(args.num_classes)],
        [f"E{i}" for i in range(label_avg.shape[1])],
        "All target subjects: mean router weight by label",
        outputs[-1],
    )
    outputs.append(os.path.join(output_dir, "all_labels_expert_top1_rate.png"))
    plot_heatmap(
        label_top1,
        [f"class {i}" for i in range(args.num_classes)],
        [f"E{i}" for i in range(label_top1.shape[1])],
        "All target subjects: top-1 expert rate by label",
        outputs[-1],
    )
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="results/seediv_sample_style_sparse_moge_loso")
    parser.add_argument("--output_dir", default="results/seediv_router_maps")
    parser.add_argument("--num_subjects", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    outputs, _, _ = per_subject_label_maps(args, output_dir)
    outputs.extend(plot_subject_expert_summary(args, output_dir))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
