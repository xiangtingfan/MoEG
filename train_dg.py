import argparse
import csv
import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from data.eeg_dataset import build_datasets_for_loso
from losses.router_losses import (
    compute_mmd_loss,
    compute_subject_cmmd_loss,
    router_balance_loss,
    router_entropy_loss,
    router_importance_loss,
    router_load_loss,
)
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


class SubjectClassBalancedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        subjects_per_batch,
        classes_per_subject,
        samples_per_class,
        num_classes,
        seed=0,
    ):
        self.dataset = dataset
        self.subjects_per_batch = int(subjects_per_batch)
        self.classes_per_subject = int(classes_per_subject)
        self.samples_per_class = int(samples_per_class)
        self.num_classes = int(num_classes)
        self.rng = np.random.default_rng(seed)

        subject_ids = dataset.subject_ids.numpy()
        labels = dataset.y.numpy()
        self.subjects = sorted(np.unique(subject_ids).astype(int).tolist())
        self.classes = list(range(self.num_classes))
        self.indices = {}
        for subject in self.subjects:
            for label in self.classes:
                mask = (subject_ids == subject) & (labels == label)
                self.indices[(subject, label)] = np.flatnonzero(mask).astype(np.int64)

        self.batch_size = self.subjects_per_batch * self.classes_per_subject * self.samples_per_class
        self.num_batches = max(1, math.ceil(len(dataset) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def _choice(self, values, size, replace=False):
        values = list(values)
        if len(values) == 0:
            return []
        replace = replace or len(values) < size
        return self.rng.choice(values, size=size, replace=replace).tolist()

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            selected_subjects = self._choice(
                self.subjects,
                self.subjects_per_batch,
                replace=len(self.subjects) < self.subjects_per_batch,
            )
            for subject in selected_subjects:
                nonempty_classes = [
                    label for label in self.classes
                    if len(self.indices[(subject, label)]) > 0
                ]
                selected_classes = self._choice(
                    nonempty_classes,
                    min(self.classes_per_subject, len(nonempty_classes)),
                    replace=len(nonempty_classes) < self.classes_per_subject,
                )
                for label in selected_classes:
                    candidates = self.indices[(subject, label)]
                    replace = len(candidates) < self.samples_per_class
                    sampled = self.rng.choice(
                        candidates,
                        size=self.samples_per_class,
                        replace=replace,
                    )
                    batch.extend(sampled.astype(int).tolist())
            self.rng.shuffle(batch)
            yield batch


def train_one_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_importance = 0.0
    total_loss_load = 0.0
    total_loss_mmd = 0.0
    total_loss_cmmd = 0.0
    total_loss_balance = 0.0
    total_loss_entropy = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        subject_ids = batch["subject_id"].to(device, non_blocking=True)
        assert_eeg_shape(x)

        optimizer.zero_grad(set_to_none=True)
        logits, router_weights_all, features = model(x, return_router=True, return_features=True)
        loss_cls = criterion(logits, y)
        loss_mmd = loss_cls.new_tensor(0.0)
        loss_cmmd = loss_cls.new_tensor(0.0)
        if args.lambda_mmd != 0:
            loss_mmd = compute_mmd_loss(
                features,
                subject_ids,
                kernel_mul=args.cmmd_kernel_mul,
                kernel_num=args.cmmd_kernel_num,
            ).to(device)
        if args.lambda_cmmd != 0:
            loss_cmmd = compute_subject_cmmd_loss(
                features,
                y,
                subject_ids,
                args.num_classes,
                kernel_mul=args.cmmd_kernel_mul,
                kernel_num=args.cmmd_kernel_num,
            ).to(device)
        loss_importance = loss_cls.new_tensor(0.0)
        loss_load = loss_cls.new_tensor(0.0)
        loss_balance = loss_cls.new_tensor(0.0)
        loss_entropy = loss_cls.new_tensor(0.0)
        if args.router_loss_type == "legacy":
            loss_balance = router_balance_loss(router_weights_all).to(device)
            loss_entropy = router_entropy_loss(router_weights_all).to(device)
            loss_router = args.lambda_balance * loss_balance + args.lambda_entropy * loss_entropy
        else:
            loss_importance = router_importance_loss(router_weights_all).to(device)
            loss_load = router_load_loss(router_weights_all).to(device)
            loss_router = args.lambda_router * (loss_importance + loss_load)
        loss = (
            loss_cls
            + loss_router
            + args.lambda_mmd * loss_mmd
            + args.lambda_cmmd * loss_cmmd
        )
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        preds = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * batch_size
        total_loss_cls += float(loss_cls.item()) * batch_size
        total_loss_importance += float(loss_importance.item()) * batch_size
        total_loss_load += float(loss_load.item()) * batch_size
        total_loss_mmd += float(loss_mmd.item()) * batch_size
        total_loss_cmmd += float(loss_cmmd.item()) * batch_size
        total_loss_balance += float(loss_balance.item()) * batch_size
        total_loss_entropy += float(loss_entropy.item()) * batch_size
        total_correct += int((preds == y).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "loss_cls": total_loss_cls / max(total_samples, 1),
        "loss_importance": total_loss_importance / max(total_samples, 1),
        "loss_load": total_loss_load / max(total_samples, 1),
        "loss_mmd": total_loss_mmd / max(total_samples, 1),
        "loss_cmmd": total_loss_cmmd / max(total_samples, 1),
        "loss_balance_legacy": total_loss_balance / max(total_samples, 1),
        "loss_entropy_legacy": total_loss_entropy / max(total_samples, 1),
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

    if args.source_sampling == "balanced":
        source_batch_sampler = SubjectClassBalancedBatchSampler(
            source_dataset,
            subjects_per_batch=args.subjects_per_batch,
            classes_per_subject=args.classes_per_subject,
            samples_per_class=args.samples_per_class,
            num_classes=args.num_classes,
            seed=args.seed + target_subject,
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=source_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
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
    history = []

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

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "loss_cls": train_metrics["loss_cls"],
            "loss_importance": train_metrics["loss_importance"],
            "loss_load": train_metrics["loss_load"],
            "loss_mmd": train_metrics["loss_mmd"],
            "loss_cmmd": train_metrics["loss_cmmd"],
            "loss_balance_legacy": train_metrics["loss_balance_legacy"],
            "loss_entropy_legacy": train_metrics["loss_entropy_legacy"],
            "train_acc": train_metrics["acc"],
            "target_acc": target_metrics["acc"],
            "target_macro_f1": target_metrics["macro_f1"],
            "target_kappa": target_metrics["kappa"],
            "best_target_acc": best["acc"],
        }
        history.append(epoch_record)

        print(
            "epoch={epoch} train_loss={train_loss:.6f} "
            "loss_cls={loss_cls:.6f} loss_importance={loss_importance:.6f} "
            "loss_load={loss_load:.6f} loss_mmd={loss_mmd:.6f} loss_cmmd={loss_cmmd:.6f} "
            "loss_balance_legacy={loss_balance_legacy:.6f} "
            "loss_entropy_legacy={loss_entropy_legacy:.6f} train_acc={train_acc:.6f} "
            "target_acc={target_acc:.6f} target_macro_f1={target_macro_f1:.6f} "
            "target_kappa={target_kappa:.6f} best_target_acc={best_target_acc:.6f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                loss_cls=train_metrics["loss_cls"],
                loss_importance=train_metrics["loss_importance"],
                loss_load=train_metrics["loss_load"],
                loss_mmd=train_metrics["loss_mmd"],
                loss_cmmd=train_metrics["loss_cmmd"],
                loss_balance_legacy=train_metrics["loss_balance_legacy"],
                loss_entropy_legacy=train_metrics["loss_entropy_legacy"],
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
    save_json(history, os.path.join(target_dir, "history.json"))
    history_csv_path = os.path.join(target_dir, "history.csv")
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else [])
        writer.writeheader()
        writer.writerows(history)

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
    parser.add_argument("--dataset_name", type=str, choices=["seed", "seediv", "seedv"], required=True)
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
    parser.add_argument("--lambda_router", type=float, default=0.01)
    parser.add_argument("--lambda_mmd", type=float, default=0.0)
    parser.add_argument("--lambda_cmmd", type=float, default=0.1)
    parser.add_argument("--cmmd_kernel_mul", type=float, default=2.0)
    parser.add_argument("--cmmd_kernel_num", type=int, default=5)
    parser.add_argument("--router_loss_type", type=str, choices=["classic", "legacy"], default="classic")
    parser.add_argument("--source_sampling", type=str, choices=["random", "balanced"], default="random")
    parser.add_argument("--subjects_per_batch", type=int, default=4)
    parser.add_argument("--classes_per_subject", type=int, default=4)
    parser.add_argument("--samples_per_class", type=int, default=8)
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
