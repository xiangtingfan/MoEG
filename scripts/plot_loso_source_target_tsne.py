import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.eeg_dataset import build_datasets_for_loso
from train_dg import build_model, resolve_device


SEEDIV_EMOTIONS = {
    0: "Neutral",
    1: "Sad",
    2: "Fear",
    3: "Happy",
}

SEED_EMOTIONS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def load_config(run_dir, target_subject, device):
    metrics_path = os.path.join(run_dir, f"target_{target_subject:02d}", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    config = payload["config"]
    config["device"] = device
    return SimpleNamespace(**config)


@torch.no_grad()
def collect_features(model, loader, device, domain_id):
    features, labels, subjects, domains = [], [], [], []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        logits, feat = model(x, return_features=True)
        _ = logits
        features.append(feat.detach().cpu().numpy())
        labels.append(batch["y"].numpy())
        subjects.append(batch["subject_id"].numpy())
        domains.append(np.full(batch["y"].shape[0], domain_id, dtype=np.int64))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(subjects, axis=0),
        np.concatenate(domains, axis=0),
    )


def maybe_subsample(features, labels, subjects, domains, max_points, seed):
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, labels, subjects, domains

    rng = np.random.default_rng(seed)
    keep = []
    groups = []
    for domain in sorted(np.unique(domains).tolist()):
        for label in sorted(np.unique(labels).tolist()):
            idx = np.flatnonzero((domains == domain) & (labels == label))
            if idx.size > 0:
                groups.append(idx)

    base = max_points // max(len(groups), 1)
    remainder = max_points - base * len(groups)
    for group_id, idx in enumerate(groups):
        take = base + (1 if group_id < remainder else 0)
        take = min(take, idx.size)
        keep.append(rng.choice(idx, size=take, replace=False))

    keep = np.concatenate(keep, axis=0)
    rng.shuffle(keep)
    return features[keep], labels[keep], subjects[keep], domains[keep]


def run_tsne(features, seed, pca_dim=30, perplexity=30):
    features = np.asarray(features, dtype=np.float32)
    np.nan_to_num(features, copy=False)
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    kwargs = dict(
        n_components=2,
        perplexity=min(perplexity, max(5, (features.shape[0] - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
        method="barnes_hut",
        angle=0.8,
        verbose=0,
    )
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(features)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(features)


def class_names(dataset_name, num_classes):
    mapping = SEED_EMOTIONS if dataset_name.lower() in {"seed", "seed3"} else SEEDIV_EMOTIONS
    return {label: mapping.get(label, f"Class {label}") for label in range(num_classes)}


def plot_source_target_tsne(
    embedding,
    labels,
    domains,
    target_subject,
    source_subjects,
    dataset_name,
    num_classes,
    output_path,
):
    names = class_names(dataset_name, num_classes)
    cmap = plt.get_cmap("tab10", max(num_classes, 1))
    markers = {0: "^", 1: "*"}
    domain_names = {0: "Source", 1: "Target"}

    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=180)
    for label in range(num_classes):
        color = cmap(label)
        for domain in [0, 1]:
            mask = (labels == label) & (domains == domain)
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=16 if domain == 0 else 42,
                alpha=0.58 if domain == 0 else 0.82,
                marker=markers[domain],
                color=color,
                edgecolors="#222222" if domain == 1 else "none",
                linewidths=0.35 if domain == 1 else 0,
                label=f"{domain_names[domain]} {label}: {names[label]}",
            )

    ax.set_title(
        f"Target S{target_subject:02d}: source vs target pre-classifier features",
        fontsize=13,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#e8e8e8", linewidth=0.7)
    ax.text(
        0.01,
        0.01,
        f"Source subjects: {', '.join(f'S{s:02d}' for s in source_subjects)}\n"
        "Same color = same label; triangle = source; star = target",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        va="bottom",
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=1.25,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_one_target(args, target_subject, device):
    model_args = load_config(args.run_dir, target_subject, args.device)
    source_dataset, target_dataset, source_subjects = build_datasets_for_loso(model_args, target_subject)

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(model_args).to(device)
    checkpoint_path = os.path.join(args.run_dir, f"target_{target_subject:02d}", "best_model.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    source = collect_features(model, source_loader, device, domain_id=0)
    target = collect_features(model, target_loader, device, domain_id=1)

    features = np.concatenate([source[0], target[0]], axis=0)
    labels = np.concatenate([source[1], target[1]], axis=0)
    subjects = np.concatenate([source[2], target[2]], axis=0)
    domains = np.concatenate([source[3], target[3]], axis=0)
    total_before = int(features.shape[0])

    features, labels, subjects, domains = maybe_subsample(
        features,
        labels,
        subjects,
        domains,
        max_points=args.max_points,
        seed=args.seed + target_subject,
    )

    embedding = run_tsne(
        features,
        seed=args.seed + target_subject,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
    )

    target_dir = os.path.join(args.run_dir, f"target_{target_subject:02d}")
    output_path = os.path.join(target_dir, args.output_name)
    plot_source_target_tsne(
        embedding=embedding,
        labels=labels,
        domains=domains,
        target_subject=target_subject,
        source_subjects=source_subjects,
        dataset_name=model_args.dataset_name,
        num_classes=model_args.num_classes,
        output_path=output_path,
    )

    np.savez(
        output_path.replace(".png", ".npz"),
        embedding=embedding,
        features=features,
        labels=labels,
        subject_ids=subjects,
        domains=domains,
        domain_names=np.array(["source", "target"]),
        total_points_before_sampling=total_before,
    )
    return {
        "target_subject": target_subject,
        "source_samples": int(source[0].shape[0]),
        "target_samples": int(target[0].shape[0]),
        "points_plotted": int(features.shape[0]),
        "output": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-target LOSO source/target t-SNE using pre-classifier features."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--start_target", type=int, default=0)
    parser.add_argument("--end_target", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=0, help="0 means plot all source+target samples.")
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--output_name", default="source_target_preclassifier_tsne.png")
    args = parser.parse_args()

    device = resolve_device(args.device)
    target_dirs = [
        name for name in os.listdir(args.run_dir)
        if name.startswith("target_") and os.path.isdir(os.path.join(args.run_dir, name))
    ]
    target_ids = sorted(int(name.split("_")[-1]) for name in target_dirs)
    if args.end_target is not None:
        target_ids = [t for t in target_ids if args.start_target <= t <= args.end_target]
    else:
        target_ids = [t for t in target_ids if t >= args.start_target]

    outputs = [make_one_target(args, target_id, device) for target_id in target_ids]
    print(json.dumps({"outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
