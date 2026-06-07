import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.eeg_dataset import build_datasets_for_loso
from train_dg import build_model, resolve_device


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_tsne(features, seed, pca_dim=30):
    features = np.asarray(features, dtype=np.float32)
    np.nan_to_num(features, copy=False)
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    perplexity = min(30, max(2, (features.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1000,
        method="barnes_hut",
        angle=0.8,
        verbose=1,
    ).fit_transform(features)


def model_args(args):
    return SimpleNamespace(
        model_name="sample_style_sparse_moge",
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        session=args.session,
        target_subject=0,
        num_subjects=args.num_subjects,
        num_classes=args.num_classes,
        time_window=args.time_window,
        stride=args.stride,
        normalization="minmax",
        batch_size=64,
        epochs=100,
        lr=0.001,
        weight_decay=0.0001,
        hidden_channels=64,
        num_layers=1,
        heads=2,
        dim_head=4,
        num_experts=4,
        top_k=2,
        temperature=1.0,
        pool="cls",
        dropout=0.2,
        nonnegative_adjacency=False,
        lambda_balance=0.05,
        lambda_entropy=0.01,
        seed=2025,
        device=args.device,
        save_dir=args.run_dir,
        num_workers=0,
    )


def router_clusters(stats_path, num_clusters):
    stats = np.load(stats_path)
    avg_weights = stats["avg_weights"]
    acc = stats["accuracy"] * 100.0
    clusters = fcluster(linkage(avg_weights, method="ward"), t=num_clusters, criterion="maxclust")
    seen = []
    for subject in range(len(clusters)):
        if clusters[subject] not in seen:
            seen.append(clusters[subject])
    remap = {old: idx + 1 for idx, old in enumerate(seen)}
    clusters = np.asarray([remap[c] for c in clusters])
    return clusters, acc, avg_weights


@torch.no_grad()
def collect_features(args):
    device = resolve_device(args.device)
    base_args = model_args(args)
    pre_expert_features = []
    final_features = []
    labels = []
    subject_ids = []

    for target in range(args.num_subjects):
        print(f"extracting target {target:02d}", flush=True)
        _, target_dataset, _ = build_datasets_for_loso(base_args, target)
        loader = DataLoader(target_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
        model = build_model(base_args).to(device)
        checkpoint = os.path.join(args.run_dir, f"target_{target:02d}", "best_model.pt")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        captured_pre_expert = []
        captured_final = []

        def capture_router_input(module, module_input, module_output):
            y = module_input[0].detach().cpu()
            captured_pre_expert.append(y.mean(dim=(1, 2)))

        def capture_fc_input(module, module_input):
            captured_final.append(module_input[0].detach().cpu())

        router_handle = model.layers[0].router.register_forward_hook(capture_router_input)
        fc_handle = model.fc.register_forward_pre_hook(capture_fc_input)

        target_labels = []
        target_subject_ids = []
        for batch in loader:
            _ = model(batch["x"].to(device))
            target_labels.append(batch["y"].numpy())
            target_subject_ids.append(batch["subject_id"].numpy())

        router_handle.remove()
        fc_handle.remove()

        pre_expert_features.append(torch.cat(captured_pre_expert, dim=0).numpy())
        final_features.append(torch.cat(captured_final, dim=0).numpy())
        labels.append(np.concatenate(target_labels, axis=0))
        subject_ids.append(np.concatenate(target_subject_ids, axis=0))

    return {
        "pre_expert": np.concatenate(pre_expert_features, axis=0),
        "final": np.concatenate(final_features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "subject_ids": np.concatenate(subject_ids, axis=0),
    }


def plot_embedding(embedding, subject_ids, clusters, acc, title, output_path):
    colors = ["#7F8F84", "#B88C8C", "#8A9199", "#B7A99A", "#9CA986", "#C4A69D"]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
    for cluster_id in sorted(np.unique(clusters).tolist()):
        subjects = np.where(clusters == cluster_id)[0]
        mask = np.isin(subject_ids, subjects)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=6,
            alpha=0.38,
            color=colors[(cluster_id - 1) % len(colors)],
            label=f"C{cluster_id}: " + ",".join(f"S{s:02d}" for s in subjects),
            linewidths=0,
        )

    for subject in range(len(clusters)):
        pts = embedding[subject_ids == subject]
        center = pts.mean(axis=0)
        ax.scatter(
            [center[0]],
            [center[1]],
            s=90,
            color=colors[(clusters[subject] - 1) % len(colors)],
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        ax.text(
            center[0] + 0.8,
            center[1] + 0.8,
            f"S{subject:02d}\nC{clusters[subject]}\n{acc[subject]:.1f}%",
            fontsize=7,
            zorder=6,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_centroid_similarity(features, subject_ids, clusters, acc, title, output_path):
    subject_features = np.asarray([features[subject_ids == s].mean(axis=0) for s in range(len(clusters))])
    if subject_features.shape[1] > 10:
        subject_features = PCA(n_components=min(10, subject_features.shape[0] - 1), random_state=2025).fit_transform(subject_features)
    subject_xy = run_tsne(subject_features, 2025, pca_dim=10) if subject_features.shape[0] > 5 else subject_features[:, :2]

    colors = ["#7F8F84", "#B88C8C", "#8A9199", "#B7A99A", "#9CA986", "#C4A69D"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    for cluster_id in sorted(np.unique(clusters).tolist()):
        subjects = np.where(clusters == cluster_id)[0]
        pts = subject_xy[subjects]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=130,
            color=colors[(cluster_id - 1) % len(colors)],
            edgecolor="black",
            linewidth=0.9,
            label=f"C{cluster_id}",
        )
        if len(subjects) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=colors[(cluster_id - 1) % len(colors)], alpha=0.4)
    for subject, (x, y) in enumerate(subject_xy):
        ax.text(x + 0.03, y + 0.03, f"S{subject:02d}\n{acc[subject]:.1f}%", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("subject feature t-SNE 1")
    ax.set_ylabel("subject feature t-SNE 2")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="seediv")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--num_subjects", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--time_window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--run_dir", default="results/seediv_sample_style_sparse_moge_loso")
    parser.add_argument("--router_stats", default="results/seediv_router_subject_analysis/subject_router_stats.npz")
    parser.add_argument("--output_dir", default="results/seediv_feature_router_relation")
    parser.add_argument("--num_clusters", type=int, default=4)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    clusters, acc, avg_weights = router_clusters(args.router_stats, args.num_clusters)
    cache_path = os.path.join(output_dir, "feature_router_relation_data.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        data = {
            "pre_expert": cached["pre_expert"],
            "final": cached["final"],
            "labels": cached["labels"],
            "subject_ids": cached["subject_ids"],
        }
    else:
        data = collect_features(args)
        np.savez(cache_path, clusters=clusters, acc=acc, **data)

    outputs = []
    for key, title in [
        ("pre_expert", "Pre-expert router-style features annotated by router clusters"),
        ("final", "Final pre-classifier features annotated by router clusters"),
    ]:
        embedding = run_tsne(data[key], 2025)
        out = os.path.join(output_dir, f"{key}_tsne_by_router_clusters.png")
        plot_embedding(embedding, data["subject_ids"], clusters, acc, title, out)
        outputs.append(out)

        out_centroid = os.path.join(output_dir, f"{key}_subject_centroids_by_router_clusters.png")
        plot_centroid_similarity(data[key], data["subject_ids"], clusters, acc, title + " (subject centroids)", out_centroid)
        outputs.append(out_centroid)

    print("\n".join(outputs))


if __name__ == "__main__":
    main()
