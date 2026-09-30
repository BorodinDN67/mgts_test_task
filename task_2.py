import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from scipy.signal import correlate, correlation_lags

DATA_PATH = './2'

fc = 20 * 10**3 #гц
det1 = np.load(DATA_PATH + '/det1.npy')
det2 = np.load(DATA_PATH + '/det2.npy')

corr = correlate(det2, det1, mode="full", method="fft")
lags = correlation_lags(len(det2), len(det1), mode="full")

lag_samples = lags[np.argmax(corr)]
delay_seconds = lag_samples / fc

n = min(len(det1), len(det2) - lag_samples)

x = det1[:n]
y = det2[lag_samples:lag_samples + n]

_, Sxx = welch(x, fc, nperseg=4096, noverlap=2048, window='hann')
_, Sxy = csd(x, y, fc, nperseg=4096, noverlap=2048, window='hann')
fig = plt.figure(figsize=(25,10))
line = np.linspace(0, 10000, len(Sxx))
plt.plot(line, np.abs(Sxy) / np.abs(Sxx))
plt.xlabel('Гц')
plt.ylabel('АЧХ')
fig.savefig("./results/AFC_finish", dpi=200, bbox_inches="tight")

print('====================================================')
print(f'Ответ на вопрос 1: {delay_seconds:.1f}')
print('Ответ на вопрос 2 сохранен в results/AFC_finish')