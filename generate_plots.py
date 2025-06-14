import matplotlib
matplotlib.use('agg')  # use the Anti-Grain Geometry backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation


# Load your W&B export
df = pd.read_csv('wandb_export_2025-06-08T05_48_06.350+12_00.csv')

# We're assuming the "stalk dim d" is stored in the 'd' column.
df = df[['dataset', 'model', 'd', 'time_elapsed', 'time_elapsed_std']]

# Mapping from raw model names to the desired labels:
def pretty_label(model_name):
    if model_name == 'BundleSheaf':
        return 'O(d)-NSD'
    elif model_name.startswith('SampleBundleSheaf_'):
        k = model_name.split('_')[-1]
        return f'O(d)-SS (k={k})'
    else:
        return model_name

datasets = df['dataset'].unique()

# No shared y-axis so each plot auto-scales
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(datasets),
    figsize=(5 * len(datasets), 4),
    sharey=False
)

if not isinstance(axes, (np.ndarray, list)):
    axes = [axes]

for ax, ds in zip(axes, datasets):
    sub = df[df['dataset'] == ds]
    dims = sorted(sub['d'].unique())

    for model in sorted(sub['model'].unique()):
        grp = sub[sub['model'] == model]
        ix = np.argsort(grp['d'].values)
        x = grp['d'].values[ix]
        y = grp['time_elapsed'].values[ix]
        yerr = grp['time_elapsed_std'].values[ix]

        label = pretty_label(model)
        ax.plot(x, y, marker='o', label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    # log base 2 so powers-of-2 are evenly spaced
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    ax.yaxis.set_major_locator(LogLocator(
        base=10.0,
        subs=range(1,10),   # will place ticks at 1,2,5 × each power of 10
        numticks=10
    ))
    # -- install minor ticks at every 1–9 × 10^n --------------
    ax.yaxis.set_minor_locator(LogLocator(
        base=10.0,
        subs=range(1,10),  # 1,2,…9 × each decade
        numticks=10
    ))
    # -- render them as plain decimals, not 10^x ----------
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    # -- turn on the minor tick marks ----------------------
    ax.minorticks_on()

    # force ticks at your exact dims and plain formatting
    ax.set_xticks(dims)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=len(dims)))

    ax.set_xlabel('Stalk dim $d$ (log₂ scale)')
    ax.set_title(ds)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Y-label and legend on the first subplot
axes[0].set_ylabel('Time (s, log scale)')
axes[0].legend(loc='best', fontsize='small')

plt.tight_layout()
plt.savefig('time_vs_dim.png', dpi=300)
