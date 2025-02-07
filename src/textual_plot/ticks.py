from math import ceil, floor, log10

import numpy as np

x_min = -11
x_max = 6

delta_x = x_max - x_min
tick_spacing = delta_x / 5
print(tick_spacing)
power = floor(log10(tick_spacing))
approx_interval = tick_spacing / 10**power

intervals = np.array([1, 2, 5, 10])
idx = np.abs(intervals - approx_interval).argmin()
interval = intervals[idx] * 10**power
print(interval)
ticks = [
    float(t * interval) for t in np.arange(ceil(x_min / interval), x_max / interval + 1)
]

print(ticks)
