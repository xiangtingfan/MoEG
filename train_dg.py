import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.eeg_dataset import build_datasets_for_loso
from losses.router_losses import router_balance_loss, router_entropy_loss
from models.sample_style_sparse_moge import SampleStyleSparseMoGE
from models.shared_residual_moe import SharedResidualMoGE
from utils.io_utils import ensure_dir, save_json, save_numpy
from utils.metrics import compute_metrics
from utils.seed import set_seed


def assert_eeg_shape(x):
    if x.ndim != 4 or x.shape[2] != 5 or x.shape[3] != 62:
        raise ValueError("Expected batch x shape [N, T, 5, 62], got {}".format(tuple(x.shape)))


def resolve_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def train_one_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        assert_eeg_shape(x)

        optimizer.zero_grad(set_to_none=True)
        logits, router_weights_all = model(x, return_router=True)
        loss_cls = criterion(logits, y)
        loss_balance = router_balance_loss(router_weights_all).to(device)
        loss_entropy = router_entropy_loss(router_weights_all).to(device)
        loss = loss_cls + args.lambda_balance * loss_balance + args.lambda_entropy * loss_entropy
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        preds = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == y).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    router_weights, subject_ids, trial_ids = [], [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        assert_eeg_shape(x)

        logits, router_weights_all = model(x, return_router=True)
        preds = logits.argmax(dim=-1)

        # router_weights_all: list of [N, E] -> [N, num_layers, E]
        if len(router_weights_all) > 0:
            stacked_router = torch.stack(router_weights_all, dim=1)
        else:
            stacked_router = torch.empty((x.size(0), 0, 0), device=x.device)

        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        router_weights.append(stacked_router.cpu().numpy())
        subject_ids.append(batch["subject_id"].cpu().numpy())
        trial_ids.append(batch["trial_id"].cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    router_info = {
        "router_weights": np.concatenate(router_weights, axis=0),
        "subject_ids": np.concatenate(subject_ids, axis=0),
        "trial_ids": np.concatenate(trial_ids, axis=0),
        "labels": y_true,
        "preds": y_pred,
    }
    return metrics, router_info


def build_model(args):
    common_kwargs = dict(
        in_channels=5,
        hidden_channels=args.hidden_channels,
        num_points=62,
        time_window=args.time_window,
        num_layers=args.num_layers,
        heads=args.heads,
        dim_head=args.dim_head,
        num_classes=args.num_classes,
        num_experts=args.num_experts,
        top_k=args.top_k,
        temperature=args.temperature,
        pool=args.pool,
        dropout=args.dropout,
        nonnegative_adjacency=args.nonnegative_adjacency,
        router_mode=args.router_mode,
        fixed_expert_index=args.fixed_expert_index,
    )
    if args.model_name == "sample_style_sparse_moge":
        return SampleStyleSparseMoGE(**common_kwargs)
    if args.model_name == "shared_residual_moge":
        return SharedResidualMoGE(
            **common_kwargs,
            expert_bottleneck=args.expert_bottleneck,
        )
    raise ValueError("Unknown model_name: {}".format(args.model_name))


def train_one_target(args, target_subject=None):
    if target_subject is None:
        target_subject = args.target_subject
    target_subject = int(target_subject)

    set_seed(args.seed)
    device = resolve_device(args.device)

    subjects = list(range(args.num_subjects))
    if target_subject not in subjects:
        raise ValueError("target_subject {} is outside [0, {})".format(target_subject, args.num_subjects))
    source_dataset, target_dataset, source_subjects = build_datasets_for_loso(args, target_subject)

    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    target_dir = ensure_dir(os.path.join(args.save_dir, "target_{:02d}".format(target_subject)))
    best = {
        "epoch": -1,
        "acc": -1.0,
        "macro_f1": 0.0,
        "kappa": 0.0,
        "confusion_matrix": [],
        "router_info": None,
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, source_loader, optimizer, criterion, device, args)
        target_metrics, router_info = evaluate(model, target_loader, device, args.num_classes)

        if target_metrics["acc"] > best["acc"]:
            best = {
                "epoch": epoch,
                "acc": target_metrics["acc"],
                "macro_f1": target_metrics["macro_f1"],
                "kappa": target_metrics["kappa"],
                "confusion_matrix": target_metrics["confusion_matrix"],
                "router_info": router_info,
            }
            torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pt"))

        print(
            "epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            "target_acc={target_acc:.6f} target_macro_f1={target_macro_f1:.6f} "
            "target_kappa={target_kappa:.6f} best_target_acc={best_target_acc:.6f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["acc"],
                target_acc=target_metrics["acc"],
                target_macro_f1=target_metrics["macro_f1"],
                target_kappa=target_metrics["kappa"],
                best_target_acc=best["acc"],
            ),
            flush=True,
        )

    # This is oracle model selection for preliminary analysis, not final strict DG protocol.
    metrics_json = {
        "target_subject": target_subject,
        "source_subjects": source_subjects,
        "best_epoch": best["epoch"],
        "best_target_acc": best["acc"],
        "best_target_macro_f1": best["macro_f1"],
        "best_target_kappa": best["kappa"],
        "best_confusion_matrix": best["confusion_matrix"],
        "model_selection": "oracle_target_test_acc",
        "strict_dg_protocol": False,
        "config": vars(args),
    }
    save_json(metrics_json, os.path.join(target_dir, "metrics.json"))

    router_info = best["router_info"]
    if router_info is not None:
        save_numpy(os.path.join(target_dir, "best_router_weights.npy"), router_info["router_weights"])
        save_numpy(os.path.join(target_dir, "best_router_subject_ids.npy"), router_info["subject_ids"])
        save_numpy(os.path.join(target_dir, "best_router_trial_ids.npy"), router_info["trial_ids"])
        save_numpy(os.path.join(target_dir, "best_router_labels.npy"), router_info["labels"])
        save_numpy(os.path.join(target_dir, "best_router_preds.npy"), router_info["preds"])

    return metrics_json


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Pure LOSO DG training for MoGE models")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["sample_style_sparse_moge", "shared_residual_moge"],
        default="sample_style_sparse_moge",
    )
    parser.add_argument("--dataset_name", type=str, choices=["seed", "seediv"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--target_subject", type=int, default=0)
    parser.add_argument("--num_subjects", type=int, default=15)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--time_window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["minmax", "zscore", "trial_minmax", "trial_zscore", "none"],
        default="minmax",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim_head", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pool", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--expert_bottleneck", type=int, default=None)
    parser.add_argument("--router_mode", type=str, choices=["learned", "single_expert"], default="learned")
    parser.add_argument("--fixed_expert_index", type=int, default=0)
    parser.add_argument("--nonnegative_adjacency", action="store_true")
    parser.add_argument("--lambda_balance", type=float, default=0.05)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    train_one_target(args)


if __name__ == "__main__":
    main()
