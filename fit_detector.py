from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent
data = pd.read_csv(root / "1/2026021218.csv", index_col="length")
data.columns = pd.to_datetime(
    data.columns.str.replace("results_", "", regex=False)
                .str.replace("_dts", "", regex=False),
    format="%Y_%m_%d%H_%M_%S",
)
data = data.sort_index().sort_index(axis=1)

train_end = data.columns[0].floor("h") + pd.Timedelta(minutes=15)
train = data.loc[:, data.columns < train_end]
baseline = train.median(axis=1).to_numpy()

changes = np.abs(np.diff(train.to_numpy(), axis=1))
noise_limit = np.median(np.median(changes, axis=0)) + 0.5

np.savez_compressed(
    root / "dts_detector.npz",
    meters=data.index.to_numpy(),
    baseline=baseline,
    low=-2.0,
    high=2.0,
    min_seconds=60.0,
    noise_limit=noise_limit,
    train_end=train_end.isoformat(),
)
print(f"детектор: {root / 'dts_detector.npz'}")
