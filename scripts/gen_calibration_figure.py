"""Regenerates fig_ece_reliability.png from the honest, uncorrupted calibration
comparison in data/calibration_comparison.csv and
data/calibration_reliability_seed0.csv (see scripts/calibration_comparison.py)."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path('figures')
DATA = Path('data')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'legend.fontsize': 9, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'figure.dpi': 150,
})

rel = pd.read_csv(DATA / 'calibration_reliability_seed0.csv')
summ = pd.read_csv(DATA / 'calibration_comparison.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.4, label='Perfect calibration')
col_map = {'GP': 'ece_gp', 'QK-SVM': 'ece_qk', 'rf_raw_700': 'ece_rf_raw', 'rf_iso_700': 'ece_rf_iso'}
for model, color, marker in [('GP', '#1565C0', 'o'), ('QK-SVM', '#C62828', 's'),
                              ('rf_raw_700', '#2E7D32', '^'), ('rf_iso_700', '#6A1B9A', 'D')]:
    sub = rel[rel.model == model]
    label = {'GP': 'Bayesian GP', 'QK-SVM': 'QK-SVM',
              'rf_raw_700': 'RF (raw, n=700)', 'rf_iso_700': 'RF (isotonic, n=700)'}[model]
    mean_ece = summ[col_map[model]].mean()
    ax1.plot(sub['conf'], sub['acc'], marker=marker, color=color, lw=2, markersize=7,
              label=f'{label} (ECE={mean_ece:.3f})')
ax1.set_xlabel('Predicted probability'); ax1.set_ylabel('Observed accuracy')
ax1.set_title('Reliability Diagram (mean ECE over 10 seeds)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
ax1.grid(alpha=0.2)

models = ['ece_gp', 'ece_qk', 'ece_rf_raw', 'ece_rf_platt', 'ece_rf_iso']
labels = ['Bayesian\nGP', 'QK-SVM', 'RF raw\n(n=700)', 'RF Platt\n(n=700)', 'RF isotonic\n(n=700)']
means = [summ[m].mean() for m in models]
errs = [1.96 * summ[m].std(ddof=1) / np.sqrt(len(summ)) for m in models]
colors_bar = ['#1565C0', '#C62828', '#2E7D32', '#F9A825', '#6A1B9A']
ax2.bar(labels, means, yerr=errs, capsize=4, color=colors_bar)
ax2.set_ylabel('Expected Calibration Error (ECE)')
ax2.set_title('ECE by Model (mean $\\pm$ 95% CI, 10 seeds)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.2)

fig.suptitle('Calibration: Bayesian GP vs. Quantum Kernel vs. Random Forest (matched $n=700$, 28-dim features)',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIGS / 'fig_ece_reliability.png', dpi=250, bbox_inches='tight')
plt.close(fig)
print('Done: fig_ece_reliability.png (honest, matched-n calibration comparison)')
