from math import ceil, floor, log10

import numpy as np

MAX_TICKS = 8

x_min = 0
x_max = 3177

delta_x = x_max - x_min
tick_spacing = delta_x / 5
power = floor(log10(tick_spacing))
approx_interval = tick_spacing / 10**power
intervals = np.array([1, 2, 5, 10])

idx = intervals.searchsorted(approx_interval)
interval = intervals[idx - 1] * 10**power
if delta_x // interval > MAX_TICKS:
    interval = intervals[idx] * 10**power
ticks = [
    float(t * interval)
    for t in np.arange(ceil(x_min / interval), x_max // interval + 1)
]
decimals = -min(0, power)
tick_labels = [f"{tick:.{decimals}f}" for tick in ticks]

print(f"{interval=:f}, {power=}")
print(f"{ticks=}")
print(f"{decimals=}")
print(f"{tick_labels=}")
