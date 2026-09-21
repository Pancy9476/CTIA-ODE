import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler

def build_time_features(date_series):

    dt = pd.to_datetime(date_series)

    hour = dt.dt.hour.values.astype(np.float32)
    weekday = dt.dt.weekday.values.astype(np.float32)
    day = dt.dt.day.values.astype(np.float32)
    month = dt.dt.month.values.astype(np.float32)

    features = np.stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),

            np.sin(2 * np.pi * weekday / 7.0),
            np.cos(2 * np.pi * weekday / 7.0),

            np.sin(2 * np.pi * day / 31.0),
            np.cos(2 * np.pi * day / 31.0),

            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ],
        axis=1
    )

    return features.astype(np.float32)


class MultivariateTimeSeriesDataset(Dataset):

    def __init__(
        self,
        values,
        time_features,
        seq_len,
        pred_len
    ):
        self.values = values.astype(np.float32)
        self.time_features = time_features.astype(np.float32)

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):

        return (
            len(self.values)
            - self.seq_len
            - self.pred_len
            + 1
        )

    def __getitem__(self, idx):

        # --------------------------------------------------------
        # Historical multivariate values
        # (L, M)
        # --------------------------------------------------------

        x = self.values[
            idx:
            idx + self.seq_len
        ]

        # --------------------------------------------------------
        # Historical time features
        # (L, F)
        # --------------------------------------------------------

        history_time = self.time_features[
            idx:
            idx + self.seq_len
        ]

        # --------------------------------------------------------
        # Future time features
        # (T, F)
        # --------------------------------------------------------

        future_time = self.time_features[
            idx + self.seq_len:
            idx + self.seq_len + self.pred_len
        ]

        # --------------------------------------------------------
        # Future labels
        # (T, M)
        # --------------------------------------------------------

        y = self.values[
            idx + self.seq_len:
            idx + self.seq_len + self.pred_len
        ]

        # --------------------------------------------------------
        # Convert
        #
        # x            -> (M, L)
        # history_time -> (F, L)
        # future_time  -> (F, T)
        # y            -> (M, T)
        # --------------------------------------------------------

        x = torch.from_numpy(x).transpose(0, 1)

        history_time = torch.from_numpy(
            history_time
        ).transpose(0, 1)

        future_time = torch.from_numpy(
            future_time
        ).transpose(0, 1)

        y = torch.from_numpy(y).transpose(0, 1)

        return x, history_time, future_time, y


def load_and_split_data(
    csv_path,
    seq_len,
    pred_len,
    batch_size=256,
    num_workers=0
):

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Find date column
    # ------------------------------------------------------------

    possible_date_columns = [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp",
        "Timestamp"
    ]

    date_col = None

    for col in possible_date_columns:

        if col in df.columns:

            date_col = col
            break

    if date_col is None:

        raise ValueError(
            "Not Find"
            "support date / Date / datetime / timestamp"
        )

    # ------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------

    time_features = build_time_features(
        df[date_col]
    )

    # ------------------------------------------------------------
    # Values
    # ------------------------------------------------------------

    value_df = df.drop(
        columns=[date_col]
    )

    values = value_df.values.astype(
        np.float32
    )

    num_vars = values.shape[1]

    dataset_name = os.path.basename(
        csv_path
    ).split(".")[0].lower()

    total_len = len(values)

    # ============================================================
    # Standard ETT split
    # ============================================================

    if "etth" in dataset_name:
        train_end = 12 * 30 * 24
        val_end = 16 * 30 * 24
        test_end = 20 * 30 * 24
        print(f"[{dataset_name}] use ETTh standard cut")

    elif "ettm" in dataset_name:
        train_end = 12 * 30 * 24 * 4  # 34560
        val_end = 16 * 30 * 24 * 4  # 46080
        test_end = 20 * 30 * 24 * 4  # 57600
        print(f"[{dataset_name}] use ETTm standard cut")

    else:

        train_end = int(
            total_len * 0.7
        )

        val_end = int(
            total_len * 0.8
        )

        test_end = total_len

        print(
            f"[{dataset_name}] use 70/10/20 cut"
        )

    print(
        f"Train: [0 : {train_end}]"
    )

    print(
        f"Val  : [{train_end} : {val_end}]"
    )

    print(
        f"Test : [{val_end} : {test_end}]"
    )

    # ============================================================
    # Scaling
    # ============================================================

    scaler = StandardScaler()

    scaler.fit(
        values[:train_end]
    )

    values_scaled = scaler.transform(
        values
    )

    # ============================================================
    # Train
    # ============================================================

    train_values = values_scaled[
        :train_end
    ]

    train_time = time_features[
        :train_end
    ]

    # ============================================================
    # Validation
    # ============================================================

    val_start = max(
        0,
        train_end - seq_len
    )

    val_values = values_scaled[
        val_start:
        val_end
    ]

    val_time = time_features[
        val_start:
        val_end
    ]

    # ============================================================
    # Test
    # ============================================================

    test_start = max(
        0,
        val_end - seq_len
    )

    test_values = values_scaled[
        test_start:
        test_end
    ]

    test_time = time_features[
        test_start:
        test_end
    ]

    # ============================================================
    # Dataset
    # ============================================================

    train_dataset = MultivariateTimeSeriesDataset(
        train_values,
        train_time,
        seq_len,
        pred_len
    )

    val_dataset = MultivariateTimeSeriesDataset(
        val_values,
        val_time,
        seq_len,
        pred_len
    )

    test_dataset = MultivariateTimeSeriesDataset(
        test_values,
        test_time,
        seq_len,
        pred_len
    )

    # ============================================================
    # DataLoader
    # ============================================================

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(
        f"Variables: {num_vars}"
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        num_vars,
        time_features.shape[1]
    )