import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, LogLocator

matplotlib.use('agg')  

# Load and preprocess
df = pd.read_csv('wandb_export_2025-06-08T05_48_06.350+12_00.csv')
df = df[['dataset', 'model', 'd', 'time_elapsed', 'time_elapsed_std']]

# helper for labels
def pretty_label(model_name):
    if model_name == 'BundleSheaf':
        return 'O(d)-NSD'
    elif model_name.startswith('SampleBundleSheaf_'):
        k = model_name.split('_')[-1]
        return f'O(d)-SS (k={k})'
    else:
        return model_name

# 1. Compute baseline times
#   pivot so we can look up baseline time for each (dataset, d)
base = df[df['model'] == 'BundleSheaf']\
         .set_index(['dataset','d'])['time_elapsed']\
         .rename('t_base')
df = df.join(base, on=['dataset','d'])

# 2. Compute speedup
df['speedup'] = df['t_base'] / df['time_elapsed']
# (Optional) propagate errorbars by standard error propagation:
df['speedup_std'] = df['speedup'] * (df['time_elapsed_std'] / df['time_elapsed'])

datasets = df['dataset'].unique()

# 3. Plot speedup vs d
fig, axes = plt.subplots(
    1, len(datasets),
    figsize=(5*len(datasets), 4),
    sharey=False
)
if not hasattr(axes, "__iter__"):
    axes = [axes]

for ax, ds in zip(axes, datasets):
    sub = df[df['dataset'] == ds]
    dims = sorted(sub['d'].unique())

    for model in sorted(sub['model'].unique()):
        grp = sub[sub['model'] == model]
        ix = np.argsort(grp['d'].values)
        x = grp['d'].values[ix]
        y = grp['speedup'].values[ix]
        if 'speedup_std' in grp:
            errs = grp['speedup_std'].values
        else:
            errs = np.zeros_like(grp['speedup'].values)

        yerr = errs[ix]

        ax.plot(x, y, marker='o', label=pretty_label(model))
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_xscale('log', base=2)
    ax.set_xticks(dims)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=len(dims)))

    ax.set_xlabel('Stalk dim $d$ (log₂ scale)')
    ax.set_title(ds)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

axes[0].set_ylabel('Speedup over O(d)-NSD')
axes[0].legend(loc='best', fontsize='small')

plt.tight_layout()
plt.savefig('speedup_vs_dim.png', dpi=300)
