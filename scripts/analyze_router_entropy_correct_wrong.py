import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def router_entropy(weights):
    if weights.ndim == 3:
        weights = weights.mean(axis=1)
    weights = np.clip(weights.astype(np.float64), 1e-12, 1.0)
    return -(weights * np.log(weights)).sum(axis=1)


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    # delta = P(x > y) - P(x < y). Here x=wrong, y=correct.
    values = np.concatenate([x, y])
    ranks = stats.rankdata(values, method="average")
    rank_x = ranks[: x.size].sum()
    u_x = rank_x - x.size * (x.size + 1) / 2.0
    return (2.0 * u_x) / (x.size * y.size) - 1.0


def rank_biserial_from_u(u_correct, n_correct, n_wrong):
    # scipy returns U for correct vs wrong. Positive means wrong tends higher.
    return 1.0 - (2.0 * u_correct) / (n_correct * n_wrong)


def save_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(name, "")) for name in fieldnames) + "\n")


def plot_boxplot(correct, wrong, subject_rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)

    axes[0].boxplot(
        [correct, wrong],
        labels=["Correct", "Wrong"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#8ecae6", alpha=0.8),
        medianprops=dict(color="#222222", linewidth=1.5),
    )
    axes[0].stripplot if False else None
    rng = np.random.default_rng(2025)
    for i, data in enumerate([correct, wrong], start=1):
        if data.size > 1200:
            idx = rng.choice(data.size, size=1200, replace=False)
            data = data[idx]
        jitter = rng.normal(0, 0.045, size=data.size)
        axes[0].scatter(
            np.full(data.size, i) + jitter,
            data,
            s=4,
            alpha=0.12,
            color="#2b2d42",
            linewidths=0,
        )
    axes[0].set_title("Sample-level router entropy")
    axes[0].set_ylabel("Entropy")
    axes[0].grid(True, axis="y", color="#e6e6e6", linewidth=0.7)

    subj_correct = np.array([row["correct_mean_entropy"] for row in subject_rows], dtype=float)
    subj_wrong = np.array([row["wrong_mean_entropy"] for row in subject_rows], dtype=float)
    axes[1].boxplot(
        [subj_correct, subj_wrong],
        labels=["Correct", "Wrong"],
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="#ffb703", alpha=0.65),
        medianprops=dict(color="#222222", linewidth=1.5),
    )
    for c, w in zip(subj_correct, subj_wrong):
        axes[1].plot([1, 2], [c, w], color="#6c757d", alpha=0.55, linewidth=0.9)
        axes[1].scatter([1, 2], [c, w], color=["#219ebc", "#d62828"], s=18, zorder=3)
    axes[1].set_title("Subject-level mean entropy")
    axes[1].set_ylabel("Mean entropy per subject")
    axes[1].grid(True, axis="y", color="#e6e6e6", linewidth=0.7)

    fig.tight_layout()
    combined = os.path.join(output_dir, "entropy_correct_wrong_boxplots.png")
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "router_error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    all_correct_entropy = []
    all_wrong_entropy = []
    subject_rows = []

    for name in sorted(os.listdir(args.run_dir)):
        if not name.startswith("target_"):
            continue
        target_dir = os.path.join(args.run_dir, name)
        paths = {
            "weights": os.path.join(target_dir, "best_router_weights.npy"),
            "labels": os.path.join(target_dir, "best_router_labels.npy"),
            "preds": os.path.join(target_dir, "best_router_preds.npy"),
        }
        if not all(os.path.exists(path) for path in paths.values()):
            continue

        entropy = router_entropy(np.load(paths["weights"]))
        labels = np.load(paths["labels"])
        preds = np.load(paths["preds"])
        correct_mask = preds == labels

        correct_entropy = entropy[correct_mask]
        wrong_entropy = entropy[~correct_mask]
        all_correct_entropy.append(correct_entropy)
        all_wrong_entropy.append(wrong_entropy)

        subject_rows.append(
            {
                "target_subject": int(name.split("_")[-1]),
                "correct_count": int(correct_entropy.size),
                "wrong_count": int(wrong_entropy.size),
                "correct_mean_entropy": float(correct_entropy.mean()),
                "wrong_mean_entropy": float(wrong_entropy.mean()),
                "wrong_minus_correct": float(wrong_entropy.mean() - correct_entropy.mean()),
                "correct_median_entropy": float(np.median(correct_entropy)),
                "wrong_median_entropy": float(np.median(wrong_entropy)),
            }
        )

    correct = np.concatenate(all_correct_entropy)
    wrong = np.concatenate(all_wrong_entropy)

    mann = stats.mannwhitneyu(correct, wrong, alternative="two-sided")
    cliff = cliffs_delta(wrong, correct)
    rbc = rank_biserial_from_u(mann.statistic, correct.size, wrong.size)

    subject_correct = np.array([row["correct_mean_entropy"] for row in subject_rows], dtype=float)
    subject_wrong = np.array([row["wrong_mean_entropy"] for row in subject_rows], dtype=float)
    wilcoxon = stats.wilcoxon(
        subject_wrong,
        subject_correct,
        alternative="greater",
        zero_method="wilcox",
    )
    subject_effect_r = wilcoxon.statistic / (subject_wrong.size * (subject_wrong.size + 1) / 2.0)

    sample_rows = [
        {
            "test": "mann_whitney_u",
            "group_a": "correct",
            "group_b": "wrong",
            "n_correct": int(correct.size),
            "n_wrong": int(wrong.size),
            "correct_mean": float(correct.mean()),
            "wrong_mean": float(wrong.mean()),
            "correct_median": float(np.median(correct)),
            "wrong_median": float(np.median(wrong)),
            "statistic": float(mann.statistic),
            "p_value": float(mann.pvalue),
            "cliffs_delta_wrong_vs_correct": float(cliff),
            "rank_biserial_wrong_higher": float(rbc),
        }
    ]

    subject_test_rows = [
        {
            "test": "wilcoxon_signed_rank",
            "alternative": "wrong_mean_entropy > correct_mean_entropy",
            "n_subjects": int(subject_wrong.size),
            "correct_subject_mean": float(subject_correct.mean()),
            "wrong_subject_mean": float(subject_wrong.mean()),
            "mean_paired_difference": float((subject_wrong - subject_correct).mean()),
            "median_paired_difference": float(np.median(subject_wrong - subject_correct)),
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.pvalue),
            "rank_effect_wrong_higher": float(subject_effect_r),
            "num_subjects_wrong_higher": int(np.sum(subject_wrong > subject_correct)),
            "num_subjects_correct_higher": int(np.sum(subject_wrong < subject_correct)),
        }
    ]

    save_csv(
        os.path.join(output_dir, "entropy_sample_level_mannwhitney.csv"),
        list(sample_rows[0].keys()),
        sample_rows,
    )
    save_csv(
        os.path.join(output_dir, "entropy_subject_level_wilcoxon.csv"),
        list(subject_test_rows[0].keys()),
        subject_test_rows,
    )
    save_csv(
        os.path.join(output_dir, "entropy_subject_correct_wrong_means.csv"),
        list(subject_rows[0].keys()),
        subject_rows,
    )
    plot_path = plot_boxplot(correct, wrong, subject_rows, output_dir)

    result = {
        "sample_level": sample_rows[0],
        "subject_level": subject_test_rows[0],
        "plot": plot_path,
    }
    json_path = os.path.join(output_dir, "entropy_correct_wrong_two_level_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
