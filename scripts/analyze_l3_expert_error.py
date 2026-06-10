import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEEDIV_LABEL_NAMES = {
    0: "Neutral",
    1: "Sad",
    2: "Fear",
    3: "Happy",
}


def save_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(name, "")) for name in fieldnames) + "\n")


def top1_expert(router_weights):
    if router_weights.ndim == 3:
        router_weights = router_weights.mean(axis=1)
    return router_weights.argmax(axis=1)


def load_target_arrays(run_dir, target_subject):
    target_dir = os.path.join(run_dir, f"target_{target_subject:02d}")
    return {
        "weights": np.load(os.path.join(target_dir, "best_router_weights.npy")),
        "labels": np.load(os.path.join(target_dir, "best_router_labels.npy")),
        "preds": np.load(os.path.join(target_dir, "best_router_preds.npy")),
    }


def collect_rows(run_dir, target_subjects, num_labels, num_experts, focus_label):
    subject_label_rows = []
    l3_subject_expert_rows = []
    l3_subject_summary_rows = []
    l3_global_expert_rows = []

    global_l3_counts = np.zeros(num_experts, dtype=np.int64)
    global_l3_correct = np.zeros(num_experts, dtype=np.int64)

    for subject in target_subjects:
        arrays = load_target_arrays(run_dir, subject)
        labels = arrays["labels"]
        preds = arrays["preds"]
        experts = top1_expert(arrays["weights"])
        correct = preds == labels

        for label in range(num_labels):
            mask = labels == label
            count = int(mask.sum())
            acc = float(correct[mask].mean()) if count > 0 else np.nan
            subject_label_rows.append(
                {
                    "subject": subject,
                    "label": label,
                    "label_name": SEEDIV_LABEL_NAMES.get(label, f"L{label}"),
                    "count": count,
                    "accuracy": acc,
                    "correct": int(correct[mask].sum()),
                }
            )

        l3_mask = labels == focus_label
        l3_count = int(l3_mask.sum())
        l3_acc = float(correct[l3_mask].mean()) if l3_count > 0 else np.nan
        l3_e3_fraction = np.nan

        for expert in range(num_experts):
            mask = l3_mask & (experts == expert)
            count = int(mask.sum())
            acc = float(correct[mask].mean()) if count > 0 else np.nan
            frac = float(count / l3_count) if l3_count > 0 else np.nan
            l3_subject_expert_rows.append(
                {
                    "subject": subject,
                    "label": focus_label,
                    "expert": expert,
                    "count": count,
                    "fraction": frac,
                    "accuracy": acc,
                    "correct": int(correct[mask].sum()),
                }
            )
            global_l3_counts[expert] += count
            global_l3_correct[expert] += int(correct[mask].sum())
            if expert == 3:
                l3_e3_fraction = frac

        l3_subject_summary_rows.append(
            {
                "subject": subject,
                "label": focus_label,
                "label_name": SEEDIV_LABEL_NAMES.get(focus_label, f"L{focus_label}"),
                "l3_count": l3_count,
                "l3_accuracy": l3_acc,
                "l3_top1_E3_fraction": l3_e3_fraction,
            }
        )

    for expert in range(num_experts):
        count = int(global_l3_counts[expert])
        correct = int(global_l3_correct[expert])
        l3_global_expert_rows.append(
            {
                "label": focus_label,
                "expert": expert,
                "count": count,
                "fraction": float(count / max(global_l3_counts.sum(), 1)),
                "accuracy": float(correct / count) if count > 0 else np.nan,
                "correct": correct,
            }
        )

    return (
        subject_label_rows,
        l3_subject_expert_rows,
        l3_subject_summary_rows,
        l3_global_expert_rows,
    )


def plot_subject_label_accuracy(rows, output_path, num_labels):
    subjects = sorted({int(row["subject"]) for row in rows})
    mat = np.full((len(subjects), num_labels), np.nan)
    for row in rows:
        mat[subjects.index(int(row["subject"])), int(row["label"])] = float(row["accuracy"])

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=180)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(num_labels))
    ax.set_xticklabels([f"L{i}\n{SEEDIV_LABEL_NAMES.get(i, '')}" for i in range(num_labels)])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"S{s:02d}" for s in subjects])
    ax.set_title("Subject × Label Accuracy")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j] * 100:.1f}", ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_l3_e3_scatter(rows, output_path):
    x = np.array([float(row["l3_top1_E3_fraction"]) for row in rows], dtype=float)
    y = np.array([float(row["l3_accuracy"]) for row in rows], dtype=float)
    subjects = [int(row["subject"]) for row in rows]

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=180)
    ax.scatter(x, y, s=42, color="#d62828", edgecolors="#222222", linewidths=0.5)
    for subject, xi, yi in zip(subjects, x, y):
        ax.text(xi + 0.008, yi + 0.006, f"S{subject:02d}", fontsize=8)

    if len(x) >= 2:
        coef = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, coef[0] * x_line + coef[1], color="#264653", linewidth=1.3)

    ax.set_xlabel("L3 top-1 E3 fraction")
    ax.set_ylabel("L3 accuracy")
    ax.set_title(
        "L3 accuracy vs E3 routing fraction\n"
        f"Pearson r={pearson.statistic:.3f}, p={pearson.pvalue:.3g}; "
        f"Spearman rho={spearman.statistic:.3f}, p={spearman.pvalue:.3g}"
    )
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return pearson, spearman


def plot_l3_subject_expert_heatmaps(rows, output_path, num_experts):
    subjects = sorted({int(row["subject"]) for row in rows})
    frac = np.full((len(subjects), num_experts), np.nan)
    acc = np.full((len(subjects), num_experts), np.nan)
    for row in rows:
        sidx = subjects.index(int(row["subject"]))
        eidx = int(row["expert"])
        frac[sidx, eidx] = float(row["fraction"])
        acc[sidx, eidx] = float(row["accuracy"]) if row["accuracy"] != "nan" else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=180)
    for ax, mat, title, cmap in [
        (axes[0], frac, "L3 top-1 expert fraction", "magma"),
        (axes[1], acc, "L3 accuracy within top-1 expert", "viridis"),
    ]:
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels([f"S{s:02d}" for s in subjects])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j] * 100:.1f}", ha="center", va="center", color="white", fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--focus_label", type=int, default=3)
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "router_error_analysis", "l3_expert_error")
    os.makedirs(output_dir, exist_ok=True)
    target_subjects = sorted(
        int(name.split("_")[-1])
        for name in os.listdir(args.run_dir)
        if name.startswith("target_") and os.path.isdir(os.path.join(args.run_dir, name))
    )

    subject_label_rows, l3_subject_expert_rows, l3_subject_summary_rows, l3_global_expert_rows = collect_rows(
        args.run_dir,
        target_subjects,
        args.num_labels,
        args.num_experts,
        args.focus_label,
    )

    save_csv(
        os.path.join(output_dir, "subject_label_accuracy.csv"),
        ["subject", "label", "label_name", "count", "correct", "accuracy"],
        subject_label_rows,
    )
    save_csv(
        os.path.join(output_dir, "l3_subject_expert_accuracy.csv"),
        ["subject", "label", "expert", "count", "fraction", "correct", "accuracy"],
        l3_subject_expert_rows,
    )
    save_csv(
        os.path.join(output_dir, "l3_subject_e3_fraction_vs_accuracy.csv"),
        ["subject", "label", "label_name", "l3_count", "l3_accuracy", "l3_top1_E3_fraction"],
        l3_subject_summary_rows,
    )
    save_csv(
        os.path.join(output_dir, "l3_global_expert_accuracy.csv"),
        ["label", "expert", "count", "fraction", "correct", "accuracy"],
        l3_global_expert_rows,
    )

    plot_subject_label_accuracy(
        subject_label_rows,
        os.path.join(output_dir, "subject_label_accuracy_heatmap.png"),
        args.num_labels,
    )
    pearson, spearman = plot_l3_e3_scatter(
        l3_subject_summary_rows,
        os.path.join(output_dir, "l3_e3_fraction_vs_accuracy_scatter.png"),
    )
    plot_l3_subject_expert_heatmaps(
        l3_subject_expert_rows,
        os.path.join(output_dir, "l3_subject_expert_fraction_accuracy_heatmaps.png"),
        args.num_experts,
    )

    result = {
        "focus_label": args.focus_label,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "outputs": {
            "subject_label_accuracy": os.path.join(output_dir, "subject_label_accuracy.csv"),
            "l3_subject_expert_accuracy": os.path.join(output_dir, "l3_subject_expert_accuracy.csv"),
            "l3_subject_e3_fraction_vs_accuracy": os.path.join(output_dir, "l3_subject_e3_fraction_vs_accuracy.csv"),
            "l3_global_expert_accuracy": os.path.join(output_dir, "l3_global_expert_accuracy.csv"),
        },
    }
    with open(os.path.join(output_dir, "l3_expert_error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
