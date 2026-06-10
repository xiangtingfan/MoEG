import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_NAMES = {
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


def load_subject(run_dir, subject):
    target_dir = os.path.join(run_dir, f"target_{subject:02d}")
    weights = np.load(os.path.join(target_dir, "best_router_weights.npy"))
    labels = np.load(os.path.join(target_dir, "best_router_labels.npy"))
    preds = np.load(os.path.join(target_dir, "best_router_preds.npy"))
    experts = top1_expert(weights)
    return labels, preds, experts


def collect(run_dir, subjects, num_labels, num_experts, focus_subjects, focus_labels):
    subject_label_pred_rows = []
    subject_label_expert_pred_rows = []
    focus_rows = []
    global_expert_bias_rows = []

    global_counts = {
        (label, expert, pred): 0
        for label in range(num_labels)
        for expert in range(num_experts)
        for pred in range(num_labels)
    }

    for subject in subjects:
        labels, preds, experts = load_subject(run_dir, subject)
        for label in range(num_labels):
            label_mask = labels == label
            label_count = int(label_mask.sum())
            for pred in range(num_labels):
                mask = label_mask & (preds == pred)
                count = int(mask.sum())
                subject_label_pred_rows.append(
                    {
                        "subject": subject,
                        "true_label": label,
                        "true_label_name": LABEL_NAMES.get(label, f"L{label}"),
                        "pred_label": pred,
                        "pred_label_name": LABEL_NAMES.get(pred, f"L{pred}"),
                        "count": count,
                        "fraction_within_true_label": float(count / label_count) if label_count > 0 else np.nan,
                        "is_correct_cell": int(label == pred),
                    }
                )
                if subject in focus_subjects and label in focus_labels:
                    focus_rows.append(
                        {
                            "subject": subject,
                            "true_label": label,
                            "true_label_name": LABEL_NAMES.get(label, f"L{label}"),
                            "pred_label": pred,
                            "pred_label_name": LABEL_NAMES.get(pred, f"L{pred}"),
                            "count": count,
                            "fraction_within_true_label": float(count / label_count) if label_count > 0 else np.nan,
                            "is_error": int(label != pred),
                        }
                    )

            for expert in range(num_experts):
                label_expert_mask = label_mask & (experts == expert)
                label_expert_count = int(label_expert_mask.sum())
                for pred in range(num_labels):
                    mask = label_expert_mask & (preds == pred)
                    count = int(mask.sum())
                    global_counts[(label, expert, pred)] += count
                    subject_label_expert_pred_rows.append(
                        {
                            "subject": subject,
                            "true_label": label,
                            "true_label_name": LABEL_NAMES.get(label, f"L{label}"),
                            "top1_expert": expert,
                            "pred_label": pred,
                            "pred_label_name": LABEL_NAMES.get(pred, f"L{pred}"),
                            "count": count,
                            "fraction_within_subject_label_expert": (
                                float(count / label_expert_count) if label_expert_count > 0 else np.nan
                            ),
                            "fraction_within_subject_label": float(count / label_count) if label_count > 0 else np.nan,
                            "subject_label_expert_count": label_expert_count,
                        }
                    )

    for label in range(num_labels):
        for expert in range(num_experts):
            total = sum(global_counts[(label, expert, pred)] for pred in range(num_labels))
            for pred in range(num_labels):
                count = global_counts[(label, expert, pred)]
                global_expert_bias_rows.append(
                    {
                        "true_label": label,
                        "true_label_name": LABEL_NAMES.get(label, f"L{label}"),
                        "top1_expert": expert,
                        "pred_label": pred,
                        "pred_label_name": LABEL_NAMES.get(pred, f"L{pred}"),
                        "count": count,
                        "fraction_within_label_expert": float(count / total) if total > 0 else np.nan,
                        "label_expert_count": total,
                    }
                )

    return (
        subject_label_pred_rows,
        subject_label_expert_pred_rows,
        focus_rows,
        global_expert_bias_rows,
    )


def plot_focus_heatmaps(focus_rows, focus_subjects, focus_labels, num_labels, output_path):
    fig, axes = plt.subplots(len(focus_subjects), len(focus_labels), figsize=(9.5, 9.5), dpi=180)
    if axes.ndim == 1:
        axes = axes.reshape(len(focus_subjects), len(focus_labels))

    lookup = {
        (int(row["subject"]), int(row["true_label"]), int(row["pred_label"])): float(row["fraction_within_true_label"])
        for row in focus_rows
    }
    count_lookup = {
        (int(row["subject"]), int(row["true_label"]), int(row["pred_label"])): int(row["count"])
        for row in focus_rows
    }

    for i, subject in enumerate(focus_subjects):
        for j, label in enumerate(focus_labels):
            ax = axes[i, j]
            values = np.array([lookup.get((subject, label, pred), 0.0) for pred in range(num_labels)])
            counts = np.array([count_lookup.get((subject, label, pred), 0) for pred in range(num_labels)])
            colors = ["#2a9d8f" if pred == label else "#e76f51" for pred in range(num_labels)]
            ax.bar(range(num_labels), values, color=colors, alpha=0.85)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(num_labels))
            ax.set_xticklabels([f"L{p}" for p in range(num_labels)])
            ax.set_title(f"S{subject:02d} true L{label} ({LABEL_NAMES.get(label, '')})")
            ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
            for pred, value in enumerate(values):
                ax.text(pred, value + 0.02, f"{value * 100:.1f}%\n n={counts[pred]}", ha="center", fontsize=7)
            if j == 0:
                ax.set_ylabel("Prediction fraction")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_global_label_expert_bias(rows, num_labels, num_experts, output_path):
    fig, axes = plt.subplots(num_labels, num_experts, figsize=(12.5, 10.5), dpi=180)
    lookup = {
        (int(row["true_label"]), int(row["top1_expert"]), int(row["pred_label"])): float(
            row["fraction_within_label_expert"]
        )
        for row in rows
    }
    count_lookup = {
        (int(row["true_label"]), int(row["top1_expert"]), int(row["pred_label"])): int(row["count"])
        for row in rows
    }
    for label in range(num_labels):
        for expert in range(num_experts):
            ax = axes[label, expert]
            values = np.array([lookup.get((label, expert, pred), 0.0) for pred in range(num_labels)])
            counts = np.array([count_lookup.get((label, expert, pred), 0) for pred in range(num_labels)])
            colors = ["#457b9d" if pred == label else "#f4a261" for pred in range(num_labels)]
            ax.bar(range(num_labels), values, color=colors, alpha=0.88)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(num_labels))
            ax.set_xticklabels([f"L{p}" for p in range(num_labels)], fontsize=7)
            ax.set_title(f"True L{label}, E{expert}", fontsize=8)
            ax.grid(True, axis="y", color="#ededed", linewidth=0.6)
            for pred, value in enumerate(values):
                if counts[pred] > 0:
                    ax.text(pred, value + 0.015, f"{value * 100:.0f}", ha="center", fontsize=6)
            if expert == 0:
                ax.set_ylabel(LABEL_NAMES.get(label, f"L{label}"), fontsize=8)
    fig.suptitle("Global predicted-label distribution by true label and top-1 expert", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--focus_subjects", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--focus_labels", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "router_error_analysis", "subject_label_expert_errors")
    os.makedirs(output_dir, exist_ok=True)
    subjects = sorted(
        int(name.split("_")[-1])
        for name in os.listdir(args.run_dir)
        if name.startswith("target_") and os.path.isdir(os.path.join(args.run_dir, name))
    )

    subject_label_pred_rows, subject_label_expert_pred_rows, focus_rows, global_expert_bias_rows = collect(
        args.run_dir,
        subjects,
        args.num_labels,
        args.num_experts,
        args.focus_subjects,
        args.focus_labels,
    )

    save_csv(
        os.path.join(output_dir, "subject_label_prediction_matrix.csv"),
        [
            "subject",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "count",
            "fraction_within_true_label",
            "is_correct_cell",
        ],
        subject_label_pred_rows,
    )
    save_csv(
        os.path.join(output_dir, "subject_label_expert_prediction_matrix.csv"),
        [
            "subject",
            "true_label",
            "true_label_name",
            "top1_expert",
            "pred_label",
            "pred_label_name",
            "count",
            "fraction_within_subject_label_expert",
            "fraction_within_subject_label",
            "subject_label_expert_count",
        ],
        subject_label_expert_pred_rows,
    )
    save_csv(
        os.path.join(output_dir, "focus_s01_s02_s03_l2_l3_prediction_distribution.csv"),
        [
            "subject",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "count",
            "fraction_within_true_label",
            "is_error",
        ],
        focus_rows,
    )
    save_csv(
        os.path.join(output_dir, "global_label_expert_prediction_bias.csv"),
        [
            "true_label",
            "true_label_name",
            "top1_expert",
            "pred_label",
            "pred_label_name",
            "count",
            "fraction_within_label_expert",
            "label_expert_count",
        ],
        global_expert_bias_rows,
    )

    plot_focus_heatmaps(
        focus_rows,
        args.focus_subjects,
        args.focus_labels,
        args.num_labels,
        os.path.join(output_dir, "focus_s01_s02_s03_l2_l3_prediction_distribution.png"),
    )
    plot_global_label_expert_bias(
        global_expert_bias_rows,
        args.num_labels,
        args.num_experts,
        os.path.join(output_dir, "global_label_expert_prediction_bias.png"),
    )

    summary = {
        "output_dir": output_dir,
        "focus_subjects": args.focus_subjects,
        "focus_labels": args.focus_labels,
        "files": {
            "subject_label_prediction_matrix": os.path.join(output_dir, "subject_label_prediction_matrix.csv"),
            "subject_label_expert_prediction_matrix": os.path.join(
                output_dir, "subject_label_expert_prediction_matrix.csv"
            ),
            "focus_distribution": os.path.join(output_dir, "focus_s01_s02_s03_l2_l3_prediction_distribution.csv"),
            "global_expert_bias": os.path.join(output_dir, "global_label_expert_prediction_bias.csv"),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
