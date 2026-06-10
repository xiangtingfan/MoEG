import os
import pickle
import re
from types import SimpleNamespace

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset


SEED_VIDEO_TIME = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
SEEDIV_VIDEO_TIME = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27, 54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]

SEED_LABELS = [
    [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
    [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
    [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
]

SEEDIV_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]


def canonical_dataset_name(dataset_name):
    name = dataset_name.lower()
    if name in {"seed", "seed3"}:
        return "seed"
    if name in {"seediv", "seed4", "seed-iv"}:
        return "seediv"
    if name in {"seedv", "seed5", "seed-v"}:
        return "seedv"
    raise ValueError("dataset_name must be seed, seediv, or seedv")


def get_data_path(data_path, session, dataset_name=None):
    root = data_path.format(session=session)
    if not os.path.isdir(root):
        raise FileNotFoundError("Data path does not exist: {}".format(root))
    if dataset_name is not None and canonical_dataset_name(dataset_name) == "seedv":
        feature_root = os.path.join(root, "EEG_DE_features")
        if os.path.isdir(feature_root):
            root = feature_root
        paths = []
        for filename in os.listdir(root):
            if filename.startswith(".") or not filename.lower().endswith(".npz"):
                continue
            full_path = os.path.join(root, filename)
            if os.path.isfile(full_path):
                paths.append(full_path)
        paths = sorted(paths, key=_seedv_subject_sort_key)
        if not paths:
            raise FileNotFoundError("No .npz files found in {}".format(root))
        return paths

    paths = []
    for filename in os.listdir(root):
        if filename.startswith(".") or not filename.lower().endswith(".mat"):
            continue
        full_path = os.path.join(root, filename)
        if os.path.isfile(full_path):
            paths.append(full_path)
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError("No .mat files found in {}".format(root))
    return paths


def _seedv_subject_sort_key(path):
    filename = os.path.basename(path)
    match = re.match(r"(\d+)_", filename)
    if match:
        return int(match.group(1))
    return filename


def get_trial_config(dataset_name):
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name == "seed":
        return SEED_VIDEO_TIME, SEED_LABELS
    if dataset_name == "seedv":
        return None, None
    return SEEDIV_VIDEO_TIME, SEEDIV_LABELS


def _extract_from_struct(sample, struct_key):
    if struct_key not in sample:
        return None, None
    struct_obj = sample[struct_key]
    if isinstance(struct_obj, np.ndarray) and struct_obj.dtype == np.object_ and struct_obj.size == 1:
        struct_obj = struct_obj.item()
    feature = getattr(struct_obj, "feature", None)
    label = getattr(struct_obj, "label", None)
    if feature is None or label is None:
        return None, None
    return feature, label


def _extract_feature_and_label(sample, session):
    struct_key = "dataset_session{}".format(session)
    feature, label = _extract_from_struct(sample, struct_key)
    if feature is not None and label is not None:
        return feature, label

    feature_key = struct_key + ".feature"
    label_key = struct_key + ".label"
    if feature_key in sample and label_key in sample:
        return sample[feature_key], sample[label_key]

    dynamic_feature_key = next((key for key in sample.keys() if key.endswith(".feature")), None)
    dynamic_label_key = next((key for key in sample.keys() if key.endswith(".label")), None)
    if dynamic_feature_key and dynamic_label_key:
        return sample[dynamic_feature_key], sample[dynamic_label_key]

    raise KeyError("Cannot find dataset_session{}.feature / label in mat file".format(session))


def _map_frame_labels(frame_labels, dataset_name):
    frame_labels = np.asarray(frame_labels).reshape(-1).astype(np.int64)
    if canonical_dataset_name(dataset_name) == "seed" and set(frame_labels.tolist()).issubset({-1, 0, 1}):
        frame_labels = frame_labels + 1
    return frame_labels


def _normalize_subject(features, normalization="minmax"):
    features = torch.from_numpy(features).float()
    if normalization == "none":
        return features.numpy()
    if normalization == "minmax":
        features_min = features.amin(dim=0, keepdim=True)
        features_max = features.amax(dim=0, keepdim=True)
        return ((features - features_min) / torch.clamp(features_max - features_min, min=1e-8)).numpy()
    if normalization == "zscore":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False)
        return ((features - mean) / torch.clamp(std, min=1e-8)).numpy()
    raise ValueError("normalization must be one of: minmax, zscore, trial_minmax, trial_zscore, none")


def _window_slice(trial_feature, time_window, stride):
    trial_feature = np.asarray(trial_feature).reshape(-1, 310)
    if trial_feature.shape[0] < time_window:
        return np.empty((0, time_window, 310), dtype=np.float32)
    starts = range(0, trial_feature.shape[0] - time_window + 1, stride)
    return np.stack([trial_feature[i:i + time_window] for i in starts]).astype(np.float32)


def _load_seedv_npz(path):
    sample = np.load(path, allow_pickle=True)
    if "data" not in sample or "label" not in sample:
        raise KeyError("{} must contain pickled 'data' and 'label' arrays".format(path))
    data = pickle.loads(sample["data"])
    label = pickle.loads(sample["label"])
    return data, label


def _read_seedv_expected_trial_frames(path, session):
    root = os.path.dirname(path)
    if os.path.basename(root).lower() == "eeg_de_features":
        root = os.path.dirname(root)
    timestamp_path = os.path.join(root, "trial_start_end_timestamp.txt")
    if not os.path.isfile(timestamp_path):
        return None

    with open(timestamp_path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = r"Session\s+{}:\s*start_second:\s*\[(.*?)\]\s*end_second:\s*\[(.*?)\]".format(int(session))
    match = re.search(pattern, text, flags=re.S)
    if match is None:
        return None
    starts = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
    ends = [int(item.strip()) for item in match.group(2).split(",") if item.strip()]
    if len(starts) != 15 or len(ends) != 15:
        raise ValueError("{} session {} should contain 15 start/end timestamps".format(timestamp_path, session))
    return [(end - start) // 4 for start, end in zip(starts, ends)]


def _load_seedv_subject(path, session, time_window, stride, subject_id, normalization="minmax"):
    session = int(session)
    if session not in {1, 2, 3, 123}:
        raise ValueError("SEED-V session must be 1, 2, 3, or 123, got {}".format(session))

    data, label = _load_seedv_npz(path)
    if session == 123:
        trial_indices = list(range(45))
        expected_trial_frames = []
        for single_session in [1, 2, 3]:
            frames = _read_seedv_expected_trial_frames(path, single_session)
            expected_trial_frames.extend(frames if frames is not None else [None] * 15)
    else:
        trial_indices = list(range((session - 1) * 15, session * 15))
        expected_trial_frames = _read_seedv_expected_trial_frames(path, session)

    all_trial_features = []
    for local_trial_index, trial_index in enumerate(trial_indices):
        if trial_index not in data or trial_index not in label:
            raise KeyError("{} missing SEED-V trial key {}".format(path, trial_index))
        trial_feature = np.asarray(data[trial_index], dtype=np.float32)
        if trial_feature.ndim != 2:
            trial_feature = trial_feature.reshape(trial_feature.shape[0], -1)
        if trial_feature.shape[1] != 310:
            raise ValueError("{} trial {} feature dim should be 310, got {}".format(
                path, trial_index, trial_feature.shape[1]
            ))
        expected_frame_count = None if expected_trial_frames is None else expected_trial_frames[local_trial_index]
        if expected_frame_count is not None and trial_feature.shape[0] != expected_frame_count:
            raise ValueError("{} trial {} frame count mismatch with timestamp: expected {}, got {}".format(
                path, trial_index, expected_frame_count, trial_feature.shape[0]
            ))
        all_trial_features.append(trial_feature)

    if normalization in {"minmax", "zscore", "none"}:
        subject_feature = np.concatenate(all_trial_features, axis=0)
        subject_feature = _normalize_subject(subject_feature, normalization=normalization)
        split_points = np.cumsum([item.shape[0] for item in all_trial_features])[:-1]
        all_trial_features = np.split(subject_feature, split_points, axis=0)

    xs, ys, subject_ids, trial_ids = [], [], [], []
    for local_trial_index, trial_index in enumerate(trial_indices):
        trial_feature = all_trial_features[local_trial_index]
        if normalization == "trial_minmax":
            trial_feature = _normalize_subject(trial_feature, normalization="minmax")
        elif normalization == "trial_zscore":
            trial_feature = _normalize_subject(trial_feature, normalization="zscore")

        trial_labels = _map_frame_labels(label[trial_index], "seedv")
        if trial_labels.shape[0] != trial_feature.shape[0]:
            raise ValueError("{} trial {} label frame count mismatch: feature {}, label {}".format(
                path, trial_index, trial_feature.shape[0], trial_labels.shape[0]
            ))
        unique_labels = np.unique(trial_labels)
        if unique_labels.size != 1:
            raise ValueError("{} trial {} contains non-constant labels: {}".format(
                path, trial_index, unique_labels.tolist()
            ))
        trial_label = int(unique_labels[0])

        windows = _window_slice(trial_feature, time_window, stride)
        if windows.shape[0] > 0:
            xs.append(windows.reshape(windows.shape[0], time_window, 5, 62))
            ys.append(np.full(windows.shape[0], trial_label, dtype=np.int64))
            subject_ids.append(np.full(windows.shape[0], subject_id, dtype=np.int64))
            trial_ids.append(np.full(windows.shape[0], local_trial_index, dtype=np.int64))

    if not xs:
        raise ValueError("{} produced no valid sliding windows".format(path))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(subject_ids, axis=0),
        np.concatenate(trial_ids, axis=0),
    )


def _load_subject(path, dataset_name, session, time_window, stride, subject_id, normalization="minmax"):
    if canonical_dataset_name(dataset_name) == "seedv":
        return _load_seedv_subject(path, session, time_window, stride, subject_id, normalization=normalization)

    trial_lengths, label_table = get_trial_config(dataset_name)
    expected_trial_labels = np.asarray(label_table[int(session) - 1], dtype=np.int64)

    sample = scio.loadmat(path, verify_compressed_data_integrity=False, squeeze_me=True, struct_as_record=False)
    feature, frame_labels = _extract_feature_and_label(sample, session)
    feature = np.asarray(feature)
    if feature.ndim != 2:
        feature = feature.reshape(feature.shape[0], -1)
    if feature.shape[1] != 310:
        raise ValueError("{} feature dim should be 310, got {}".format(path, feature.shape[1]))

    frame_labels = _map_frame_labels(frame_labels, dataset_name)
    total_frames = sum(trial_lengths)
    if feature.shape[0] != total_frames:
        raise ValueError("{} total frame count mismatch: expected {}, got {}".format(path, total_frames, feature.shape[0]))
    if frame_labels.shape[0] != total_frames:
        raise ValueError("{} label frame count mismatch: expected {}, got {}".format(path, total_frames, frame_labels.shape[0]))

    feature = feature.astype(np.float32)
    if normalization in {"minmax", "zscore", "none"}:
        feature = _normalize_subject(feature, normalization=normalization)
    xs, ys, subject_ids, trial_ids = [], [], [], []
    start = 0
    for trial_index, trial_len in enumerate(trial_lengths):
        end = start + trial_len
        trial_feature = feature[start:end]
        if normalization == "trial_minmax":
            trial_feature = _normalize_subject(trial_feature, normalization="minmax")
        elif normalization == "trial_zscore":
            trial_feature = _normalize_subject(trial_feature, normalization="zscore")
        trial_labels = frame_labels[start:end]
        unique_labels = np.unique(trial_labels)
        if unique_labels.size != 1:
            raise ValueError("{} trial {} contains non-constant labels: {}".format(path, trial_index, unique_labels.tolist()))
        label = int(unique_labels[0])
        expected_label = int(expected_trial_labels[trial_index])
        if label != expected_label:
            raise ValueError("{} trial {} label mismatch: expected {}, got {}".format(path, trial_index, expected_label, label))

        windows = _window_slice(trial_feature, time_window, stride)
        if windows.shape[0] > 0:
            xs.append(windows.reshape(windows.shape[0], time_window, 5, 62))
            ys.append(np.full(windows.shape[0], label, dtype=np.int64))
            subject_ids.append(np.full(windows.shape[0], subject_id, dtype=np.int64))
            trial_ids.append(np.full(windows.shape[0], trial_index, dtype=np.int64))
        start = end

    if not xs:
        raise ValueError("{} produced no valid sliding windows".format(path))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(subject_ids, axis=0),
        np.concatenate(trial_ids, axis=0),
    )


class EEGDomainDataset(Dataset):
    def __init__(self, x, y, subject_ids, trial_ids):
        if x.ndim != 4 or x.shape[2:] != (5, 62):
            raise ValueError("Expected x shape [N, T, 5, 62], got {}".format(x.shape))
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.subject_ids = torch.from_numpy(subject_ids).long()
        self.trial_ids = torch.from_numpy(trial_ids).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "y": self.y[index],
            "subject_id": self.subject_ids[index],
            "trial_id": self.trial_ids[index],
        }


def build_eeg_dataset(dataset_name, data_path, session, subjects, time_window, stride, normalization="minmax"):
    paths = get_data_path(data_path, session, dataset_name=dataset_name)
    if max(subjects) >= len(paths):
        raise IndexError("Requested subject {} but only found {} data files".format(max(subjects), len(paths)))

    arrays = [
        _load_subject(paths[subject], dataset_name, session, time_window, stride, subject, normalization=normalization)
        for subject in subjects
    ]
    x = np.concatenate([item[0] for item in arrays], axis=0)
    y = np.concatenate([item[1] for item in arrays], axis=0)
    subject_ids = np.concatenate([item[2] for item in arrays], axis=0)
    trial_ids = np.concatenate([item[3] for item in arrays], axis=0)
    return EEGDomainDataset(x, y, subject_ids, trial_ids)


def build_datasets_for_loso(args, target_subject):
    subjects = list(range(args.num_subjects))
    source_subjects = [s for s in subjects if s != target_subject]
    common = SimpleNamespace(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        session=args.session,
        time_window=args.time_window,
        stride=args.stride,
        normalization=getattr(args, "normalization", "minmax"),
    )
    source_dataset = build_eeg_dataset(
        common.dataset_name, common.data_path, common.session, source_subjects, common.time_window, common.stride,
        normalization=common.normalization
    )
    target_dataset = build_eeg_dataset(
        common.dataset_name, common.data_path, common.session, [target_subject], common.time_window, common.stride,
        normalization=common.normalization
    )
    return source_dataset, target_dataset, source_subjects
