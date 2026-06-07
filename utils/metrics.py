from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
