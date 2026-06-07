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

from data.eeg_dataset import build_datasets_for_loso, build_eeg_dataset
from train_dg import build_model, resolve_device


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_tsne(features, seed, pca_dim=30):
    features = np.asarray(features, dtype=np.float32)
    np.nan_to_num(features, copy=False)
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    return TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1000,
        method="barnes_hut",
        angle=0.8,
        verbose=1,
    ).fit_transform(features)


def plot_embedding(embedding, subject_ids, labels, title, output_path):
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels)
    unique_subjects = sorted(np.unique(subject_ids).tolist())
    cmap = plt.get_cmap("tab20", max(len(unique_subjects), 1))

    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    for idx, subject in enumerate(unique_subjects):
        mask = subject_ids == subject
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.72,
            color=cmap(idx),
            label=f"S{subject:02d}",
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.legend(ncol=3, fontsize=7, frameon=False, markerscale=1.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    label_path = output_path.replace(".png", "_by_label.png")
    label_cmap = plt.get_cmap("Set2", max(len(np.unique(labels)), 1))
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    for idx, label in enumerate(sorted(np.unique(labels).tolist())):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.72,
            color=label_cmap(idx),
            label=f"class {label}",
            linewidths=0,
        )
    ax.set_title(title + " (colored by label)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.legend(fontsize=8, frameon=False, markerscale=1.8)
    fig.tight_layout()
    fig.savefig(label_path)
    plt.close(fig)


def make_input_tsne(args, normalization, output_dir):
    dataset = build_eeg_dataset(
        args.dataset_name,
        args.data_path,
        args.session,
        list(range(args.num_subjects)),
        args.time_window,
        args.stride,
        normalization=normalization,
    )
    x = dataset.x.numpy().reshape(len(dataset), -1)
    subject_ids = dataset.subject_ids.numpy()
    labels = dataset.y.numpy()
    embedding = run_tsne(x, args.seed)
    out = os.path.join(output_dir, f"seediv_input_{normalization}_all_subjects_tsne.png")
    plot_embedding(
        embedding,
        subject_ids,
        labels,
        f"SEED-IV input features ({normalization}, all windows)",
        out,
    )
    np.savez(
        out.replace(".png", ".npz"),
        embedding=embedding,
        subject_ids=subject_ids,
        labels=labels,
    )
    return out


def load_config(config_path, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)["config"]
    config.setdefault("model_name", "sample_style_sparse_moge")
    config["device"] = args.device
    config["normalization"] = args.feature_normalization
    return SimpleNamespace(**config)


@torch.no_grad()
def make_preclassifier_tsne(args, output_dir):
    config_path = os.path.join(args.run_dir, "target_00", "metrics.json")
    model_args = load_config(config_path, args)
    device = resolve_device(args.device)

    all_features, all_subjects, all_labels = [], [], []
    for target in range(args.num_subjects):
        _, target_dataset, _ = build_datasets_for_loso(model_args, target)
        loader = DataLoader(target_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
        model = build_model(model_args).to(device)
        checkpoint = os.path.join(args.run_dir, f"target_{target:02d}", "best_model.pt")
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        captured = []

        def capture_fc_input(module, module_input):
            captured.append(module_input[0].detach().cpu())

        handle = model.fc.register_forward_pre_hook(capture_fc_input)
        labels, subject_ids = [], []
        for batch in loader:
            x = batch["x"].to(device)
            _ = model(x)
            labels.append(batch["y"].numpy())
            subject_ids.append(batch["subject_id"].numpy())
        handle.remove()
        all_features.append(torch.cat(captured, dim=0).numpy())
        all_labels.append(np.concatenate(labels, axis=0))
        all_subjects.append(np.concatenate(subject_ids, axis=0))

    features = np.concatenate(all_features, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    embedding = run_tsne(features, args.seed)
    out = os.path.join(output_dir, "seediv_preclassifier_minmax_all_subjects_tsne.png")
    plot_embedding(
        embedding,
        subject_ids,
        labels,
        "SEED-IV pre-classifier features (minmax LOSO checkpoints)",
        out,
    )
    np.savez(
        out.replace(".png", ".npz"),
        embedding=embedding,
        subject_ids=subject_ids,
        labels=labels,
        features=features,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="seediv")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--num_subjects", type=int, default=15)
    parser.add_argument("--time_window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--run_dir", default="results/seediv_sample_style_sparse_moge_loso")
    parser.add_argument("--feature_normalization", default="minmax", choices=["minmax", "zscore", "none"])
    parser.add_argument("--output_dir", default="results/seediv_tsne")
    parser.add_argument("--plot_kind", default="all", choices=["all", "raw", "minmax", "preclassifier"])
    parser.add_argument("--skip_preclassifier", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    paths = []
    if args.plot_kind in {"all", "raw"}:
        paths.append(make_input_tsne(args, "none", output_dir))
    if args.plot_kind in {"all", "minmax"}:
        paths.append(make_input_tsne(args, "minmax", output_dir))
    if args.plot_kind in {"all", "preclassifier"} and not args.skip_preclassifier:
        paths.append(make_preclassifier_tsne(args, output_dir))
    print(json.dumps({"outputs": paths}, indent=2))


if __name__ == "__main__":
    main()
