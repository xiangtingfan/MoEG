import os

import numpy as np

from train_dg import build_arg_parser, train_one_target
from utils.io_utils import ensure_dir, save_json


def main():
    parser = build_arg_parser()
    parser.description = "Run LOSO DG training for all target subjects"
    parser.add_argument("--start_target", type=int, default=0)
    parser.add_argument("--end_target", type=int, default=None)
    args = parser.parse_args()

    results = {}
    accs, macro_f1s, kappas = [], [], []
    ensure_dir(args.save_dir)

    end_target = args.num_subjects if args.end_target is None else args.end_target
    for target_subject in range(args.start_target, end_target):
        result = train_one_target(args, target_subject=target_subject)
        results[str(target_subject)] = result
        accs.append(result["best_target_acc"])
        macro_f1s.append(result["best_target_macro_f1"])
        kappas.append(result["best_target_kappa"])

    summary = {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "results": results,
    }
    save_json(summary, os.path.join(args.save_dir, "summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
