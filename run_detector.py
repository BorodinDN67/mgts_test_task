from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label


root = Path(__file__).resolve().parent
data = pd.read_csv(root / "1/2026021218.csv", index_col="length")
data.columns = pd.to_datetime(
    data.columns.str.replace("results_", "", regex=False)
                .str.replace("_dts", "", regex=False),
    format="%Y_%m_%d%H_%M_%S",
)
data = data.sort_index().sort_index(axis=1)

with np.load(root / "dts_detector.npz") as model:
    baseline = model["baseline"]
    low, high = float(model["low"]), float(model["high"])
    min_seconds = float(model["min_seconds"])
    noise_limit = float(model["noise_limit"])

diff = data.sub(baseline, axis=0)
values = diff.to_numpy()
step = data.columns.to_series().diff().dt.total_seconds().median()
min_frames = int(np.ceil(min_seconds / step)) + 1

# Каждая полоса на отдельном метре должна держаться не менее минуты.
mask = np.zeros(values.shape, dtype=np.int8)
time_link = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
for sign, candidate in [(-1, values <= low), (1, values >= high)]:
    components, _ = label(candidate, structure=time_link)
    keep = np.bincount(components.ravel()) >= min_frames
    keep[0] = False
    mask[keep[components]] = sign

changes = np.median(np.abs(np.diff(data.to_numpy(), axis=1)), axis=0)
bad = np.flatnonzero(changes > noise_limit)
first_bad = int(bad[0] + 1) if bad.size else data.shape[1]

# для графика объединяем соседние срабатывания одного знака в события.
centers = []
space_time_link = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
for sign in (-1, 1):
    components, _ = label(mask == sign, structure=space_time_link)
    for rows, cols in find_objects(components):
        centers.append(((cols.start + cols.stop - 1) / 2,
                        (rows.start + rows.stop - 1) / 2))

fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
image = ax.imshow(values, cmap="RdBu_r", vmin=-5, vmax=5,
                  origin="upper", aspect="auto", interpolation="hanning")
if first_bad < data.shape[1]:
    ax.axvspan(first_bad - 0.5, data.shape[1] - 0.5, color="gray", alpha=0.3)
if centers:
    x, y = np.array(centers).T
    ax.scatter(x, y, s=100, color="#ffd400", edgecolors="black",
               linewidths=1.5, zorder=3)

x = np.linspace(0, data.shape[1] - 1, 8, dtype=int)
y = np.linspace(0, data.shape[0] - 1, 7, dtype=int)
ax.set_xticks(x, data.columns[x].strftime("%H:%M:%S"), rotation=30, ha="right")
ax.set_yticks(y, data.index[y])
ax.set(xlabel="Время", ylabel="Метр кабеля",
       title=f"Отклонение от профиля train")
fig.colorbar(image, ax=ax, label="Отклонение, °C", extend="both")

out = root / "results"
out.mkdir(exist_ok=True)
fig.savefig(out / "dts_anomalies.png", dpi=160)
plt.close(fig)
np.savez_compressed(out / "anomaly_mask.npz", mask=mask,
                    meters=data.index.to_numpy(),
                    timestamps=data.columns.as_unit("ns").asi8,
                    quality_ok=np.arange(data.shape[1]) < first_bad)

