import argparse
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


def run_tsne(features, seed, pca_dim=30):
    features = np.asarray(features, dtype=np.float32)
    np.nan_to_num(features, copy=False)
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    return TSNE(
        n_components=2,
        perplexity=min(30, max(5, (features.shape[0] - 1) // 4)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1000,
        method="barnes_hut",
        angle=0.8,
        verbose=0,
    ).fit_transform(features)


def scatter_by_label(ax, embedding, labels, title):
    cmap = plt.get_cmap("Set2", 4)
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.78,
            color=cmap(int(label)),
            label=f"class {int(label)}",
            linewidths=0,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.tick_params(labelsize=7)


def build_input_features(args, subject, normalization):
    dataset = build_eeg_dataset(
        args.dataset_name,
        args.data_path,
        args.session,
        [subject],
        args.time_window,
        args.stride,
        normalization=normalization,
    )
    return dataset.x.numpy().reshape(len(dataset), -1), dataset.y.numpy()


def model_args_from_config(args):
    config = {
        "model_name": "sample_style_sparse_moge",
        "dataset_name": args.dataset_name,
        "data_path": args.data_path,
        "session": args.session,
        "target_subject": 0,
        "num_subjects": args.num_subjects,
        "num_classes": 4,
        "time_window": args.time_window,
        "stride": args.stride,
        "normalization": "minmax",
        "batch_size": 64,
        "epochs": 100,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "hidden_channels": 64,
        "num_layers": 1,
        "heads": 2,
        "dim_head": 4,
        "num_experts": 4,
        "top_k": 2,
        "temperature": 1.0,
        "pool": "cls",
        "dropout": 0.2,
        "nonnegative_adjacency": False,
        "lambda_balance": 0.05,
        "lambda_entropy": 0.01,
        "seed": 2025,
        "device": args.device,
        "save_dir": args.run_dir,
        "num_workers": 0,
    }
    return SimpleNamespace(**config)


@torch.no_grad()
def build_preclassifier_features(args, subject):
    device = resolve_device(args.device)
    model_args = model_args_from_config(args)
    _, target_dataset, _ = build_datasets_for_loso(model_args, subject)
    loader = DataLoader(target_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    model = build_model(model_args).to(device)
    checkpoint = os.path.join(args.run_dir, f"target_{subject:02d}", "best_model.pt")
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    captured = []

    def capture_fc_input(module, module_input):
        captured.append(module_input[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(capture_fc_input)
    labels = []
    for batch in loader:
        _ = model(batch["x"].to(device))
        labels.append(batch["y"].numpy())
    handle.remove()
    return torch.cat(captured, dim=0).numpy(), np.concatenate(labels, axis=0)


def plot_subject(args, subject, output_dir):
    raw_x, labels = build_input_features(args, subject, "none")
    minmax_x, minmax_labels = build_input_features(args, subject, "minmax")
    pre_x, pre_labels = build_preclassifier_features(args, subject)

    if not (np.array_equal(labels, minmax_labels) and np.array_equal(labels, pre_labels)):
        raise ValueError(f"Label mismatch for subject {subject}")

    raw_emb = run_tsne(raw_x, args.seed + subject)
    minmax_emb = run_tsne(minmax_x, args.seed + subject)
    pre_emb = run_tsne(pre_x, args.seed + subject)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    scatter_by_label(axes[0], raw_emb, labels, "raw input")
    scatter_by_label(axes[1], minmax_emb, labels, "min-max input")
    scatter_by_label(axes[2], pre_emb, labels, "pre-classifier")
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"SEED-IV subject {subject:02d} t-SNE comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = os.path.join(output_dir, f"subject_{subject:02d}_raw_minmax_preclassifier_tsne.png")
    fig.savefig(out)
    plt.close(fig)

    np.savez(
        out.replace(".png", ".npz"),
        raw_embedding=raw_emb,
        minmax_embedding=minmax_emb,
        preclassifier_embedding=pre_emb,
        labels=labels,
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
    parser.add_argument("--output_dir", default="results/seediv_tsne/subject_compare")
    parser.add_argument("--subject", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subjects = [args.subject] if args.subject is not None else list(range(args.num_subjects))
    outputs = []
    for subject in subjects:
        print(f"plotting subject {subject:02d}", flush=True)
        outputs.append(plot_subject(args, subject, args.output_dir))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
