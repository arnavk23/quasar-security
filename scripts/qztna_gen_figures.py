"""
Regenerates every data-driven Q-ZTNA figure from the real CSVs produced by
scripts/qztna_run_experiments.py (see data/qztna_*.csv). This replaces the
hardcoded-array figures previously produced by scripts/gen_figures.py.

fig1_framework (architecture diagram) is illustrative, not data-driven, and is
kept as-is from the original script.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qztna.simulation import NetworkFlowGenerator

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'legend.fontsize': 10, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'figure.dpi': 150,
})

FIGS = Path('figures')
FIGS.mkdir(exist_ok=True)
DATA = Path('data')

colors = {
    'ae': '#1f77b4', 'AE-SVM': '#1f77b4',
    'rbf': '#ff7f0e', 'RBF-SVM': '#ff7f0e',
    'RF': '#2ca02c', 'Random Forest': '#2ca02c',
    'XGBoost': '#d62728',
    'MLP': '#9467bd',
    'laplacian': '#8c564b', 'Laplacian-SVM': '#8c564b',
    'Poly-SVM': '#e377c2',
    'attack': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
}
ATTACK_TYPES = ["port_scan", "ddos", "exfiltration", "dns_tunneling", "arp_spoofing"]
ATTACK_LABELS = {"port_scan": "Port Scan", "ddos": "DDoS", "exfiltration": "Data Exfilt.",
                  "dns_tunneling": "DNS Tunneling", "arp_spoofing": "ARP Spoofing"}


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# =============================================================================
def fig1_framework():
    """Q-ZTNA architecture diagram, styled to match the HYQAD companion figure
    (figures/fig1_framework_hyqad.png): filled boxes with a bold title and a
    bullet-point content list, drop shadows, and labeled arrows anchored
    explicitly to box edges (never box centers) so arrows never cross through
    a box's own text."""
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 11.8); ax.set_ylim(0, 7.4); ax.axis('off')

    boxes = {
        'ingest': dict(xy=(0.3, 4.9), w=2.85, h=1.85, color='#4472C4',
                        title='NetFlow/IPFIX Ingestion',
                        lines=['Packet-header stats (16)', 'Flow-level features (24)',
                               'Behavioral features (24)', '$d = 64$ per flow']),
        'attack': dict(xy=(0.3, 2.3), w=2.85, h=1.85, color='#5B9BD5',
                        title='Attack Simulation',
                        lines=['Port Scan, DDoS', 'Data Exfiltration', 'DNS Tunneling',
                               'ARP Spoofing']),
        'kernel': dict(xy=(4.15, 3.6), w=3.05, h=1.85, color='#ED7D31',
                        title='Amplitude-Encoding',
                        lines=[r'$\hat{x} = x / \Vert x \Vert_2$', r'$K_A = (\hat{x}_i^\top \hat{x}_j)^2$',
                               'Nystrom, $m$ landmarks']),
        'svm': dict(xy=(8.15, 3.6), w=3.05, h=1.85, color='#70AD47',
                     title='Kernel SVM Classifier',
                     lines=['Linear SVM on', 'Nystrom features', 'Platt scaling']),
        'certify': dict(xy=(4.15, 0.5), w=3.05, h=1.85, color='#FFC000',
                          title='Certified Robustness',
                          lines=['Randomized smoothing', 'Clopper-Pearson bound',
                                 'certified radius $R$']),
        'defer': dict(xy=(8.15, 0.5), w=3.05, h=1.85, color='#A5A5A5',
                        title='Selective Prediction',
                        lines=['Uncertainty threshold $u$', 'coverage vs. accuracy',
                               'analyst deferral']),
    }

    def edge(key, side, frac=0.5):
        x, y, w, h = boxes[key]['xy'][0], boxes[key]['xy'][1], boxes[key]['w'], boxes[key]['h']
        if side == 'left':
            return (x, y + h * frac)
        if side == 'right':
            return (x + w, y + h * frac)
        if side == 'top':
            return (x + w * frac, y + h)
        if side == 'bottom':
            return (x + w * frac, y)

    for key, b in boxes.items():
        x, y, w, h = b['xy'][0], b['xy'][1], b['w'], b['h']
        shadow = FancyBboxPatch((x + 0.06, y - 0.06), w, h, facecolor='#000000',
                                 alpha=0.15, edgecolor='none', boxstyle="round,pad=0.04",
                                 zorder=1)
        ax.add_patch(shadow)
        rect = FancyBboxPatch((x, y), w, h, facecolor=b['color'], edgecolor='#333',
                               linewidth=1.2, boxstyle="round,pad=0.04", zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.30, b['title'], ha='center', va='top', fontsize=11.5,
                 fontweight='bold', color='white', zorder=3, wrap=True)
        for i, line in enumerate(b['lines']):
            ax.text(x + 0.20, y + h - 0.72 - 0.34 * i, f'• {line}', ha='left', va='top',
                     fontsize=9, color='white', zorder=3)

    # Explicit edge-to-edge arrows: (start_point, end_point, label, label_xy)
    arrows = [
        (edge('ingest', 'right', 0.75), edge('kernel', 'left', 0.80), 'flow vectors', (3.55, 5.55)),
        (edge('attack', 'right', 0.25), edge('kernel', 'left', 0.20), 'labeled attacks', (3.55, 3.15)),
        (edge('kernel', 'right'), edge('svm', 'left'), 'features', (7.6, 4.72)),
        (edge('svm', 'bottom', 0.28), edge('certify', 'top', 0.55), '', None),
        (edge('svm', 'bottom', 0.75), edge('defer', 'top', 0.45), '', None),
    ]
    for start, end, label, label_xy in arrows:
        ax.annotate('', xy=end, xytext=start,
                     arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.6,
                                      shrinkA=2, shrinkB=2,
                                      connectionstyle='arc3,rad=0.0'),
                     zorder=4)
        if label:
            ax.text(label_xy[0], label_xy[1], label, ha='center', va='bottom', fontsize=8.5,
                     style='italic', color='#333', zorder=5,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))

    ax.set_title('Q-ZTNA: Amplitude-Encoding Zero-Trust Anomaly Detection Pipeline',
                  fontsize=15, fontweight='bold', pad=14)
    fig.savefig(FIGS / 'fig1_framework_qztna.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig1_framework_qztna.png')


# =============================================================================
def fig2_detection_performance():
    df = pd.read_csv(DATA / 'qztna_detection_benchmark.csv')
    order = df.groupby('model')['accuracy'].mean().sort_values(ascending=False).index.tolist()
    metrics = ['accuracy', 'f1', 'auc', 'recall']
    labels = ['Accuracy', 'F1', 'AUC', 'Recall']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    w = 0.2
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        means = [df[df.model == mo][m].mean() for mo in order]
        errs = [ci95(df[df.model == mo][m]) for mo in order]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=2, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0.6, 1.02)
    ax.set_title('Detection Performance Across Models', fontsize=12, fontweight='bold')
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig2_detection_performance.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig2_detection_performance.png')


# =============================================================================
def fig3_matched_n():
    df = pd.read_csv(DATA / 'qztna_matched_n.csv')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kname, label, marker in [('ae', 'AE-SVM', 'o'), ('rbf', 'RBF-SVM', 's'), ('laplacian', 'Laplacian-SVM', '^')]:
        sub = df[df.kernel == kname].groupby('n')['accuracy'].agg(['mean', ci95]).reset_index()
        ax.errorbar(sub['n'], sub['mean'], yerr=sub['ci95'], marker=marker, capsize=3,
                    label=label, color=colors.get(kname), linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Training set size $n$'); ax.set_ylabel('Accuracy')
    ax.set_title('Matched-$n$ Kernel Comparison', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig3_matched_n.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig3_matched_n.png')


# =============================================================================
def fig4_certified_radii():
    df = pd.read_csv(DATA / 'qztna_certified_radii.csv')
    df_ok = df[~df.abstained]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ATTACK_TYPES)); w = 0.35
    for i, (label, color) in enumerate([('AE-SVM', colors['AE-SVM']), ('RBF-SVM', colors['RBF-SVM'])]):
        meds = [df_ok[(df_ok.model == label) & (df_ok.attack == a)]['radius'].median() for a in ATTACK_TYPES]
        errs = [df_ok[(df_ok.model == label) & (df_ok.attack == a)]['radius'].std() for a in ATTACK_TYPES]
        ax.bar(x + (i - 0.5) * w, meds, w, yerr=errs, capsize=3, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_TYPES], rotation=15, ha='right')
    ax.set_ylabel('Median certified $\\ell_2$ radius')
    ax.set_title('Certified Robustness by Attack Type ($\\alpha=0.05$)', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig4_certified_radii.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig4_certified_radii.png')


# =============================================================================
def fig5_sensitivity():
    df = pd.read_csv(DATA / 'qztna_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, color in [('AE-SVM', colors['AE-SVM']), ('RBF-SVM', colors['RBF-SVM'])]:
        sub = df[df.model == label].groupby('noise')['accuracy'].agg(['mean', ci95]).reset_index()
        ax.errorbar(sub['noise'], sub['mean'], yerr=sub['ci95'], marker='o', capsize=3,
                    label=label, color=color, linewidth=2)
    ax.set_xlabel('Traffic noise scale $\\sigma_{\\mathrm{noise}}$'); ax.set_ylabel('Accuracy')
    ax.set_title('Sensitivity to Traffic Noise', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig5_sensitivity.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig5_sensitivity.png')


# =============================================================================
def fig6_zeroday():
    df = pd.read_csv(DATA / 'qztna_zeroday.csv')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ATTACK_TYPES)); w = 0.35
    for i, (label, color) in enumerate([('AE-SVM', colors['AE-SVM']), ('RBF-SVM', colors['RBF-SVM'])]):
        means = [df[(df.model == label) & (df.held_out == a)]['auc'].mean() for a in ATTACK_TYPES]
        errs = [ci95(df[(df.model == label) & (df.held_out == a)]['auc']) for a in ATTACK_TYPES]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3, label=label, color=color)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='Chance')
    ax.set_xticks(x); ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_TYPES], rotation=15, ha='right')
    ax.set_ylabel('AUC on held-out attack type')
    ax.set_title('Zero-Day Attack Generalization (Leave-One-Attack-Out)', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig6_zeroday.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig6_zeroday.png')


# =============================================================================
def fig7_latency():
    df = pd.read_csv(DATA / 'qztna_latency.csv').sort_values('latency_ms')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(df['component'], df['latency_ms'], xerr=df['std_ms'], color='#4c72b0', capsize=3)
    ax.set_xlabel('Latency per sample (ms)')
    ax.set_title('Per-Sample Inference Latency (single CPU core)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig7_latency.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig7_latency.png')


# =============================================================================
def fig8_confusion():
    df = pd.read_csv(DATA / 'qztna_confusion.csv', index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(df.values * 100, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j] * 100
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    color='white' if v > 50 else 'black', fontsize=8)
    ax.set_title('AE-SVM Multi-Class Confusion Matrix (%, OvR)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='% of true class')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig8_confusion.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig8_confusion.png')


# =============================================================================
def fig9_roc():
    df = pd.read_csv(DATA / 'qztna_roc.csv')
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, cls in enumerate(df['cls'].unique()):
        sub = df[df.cls == cls]
        auc = sub['auc'].iloc[0]
        ax.plot(sub['fpr'], sub['tpr'], label=f'{cls} (AUC={auc:.3f})',
                color=colors['attack'][i % len(colors['attack'])] if cls != 'normal' else '#333')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves per Class (AE-SVM, One-vs-Rest)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig9_roc.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig9_roc.png')


# =============================================================================
def fig10_dimensionality():
    df = pd.read_csv(DATA / 'qztna_dimensionality.csv')
    piv = df.pivot_table(index='d', columns='model', values='accuracy', aggfunc='mean')
    adv = (piv['AE-SVM'] - piv['RBF-SVM']) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.6)
    ax.plot(adv.index, adv.values, marker='o', linewidth=2, color=colors['AE-SVM'])
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Feature dimensionality $d$'); ax.set_ylabel('AE-SVM $-$ RBF-SVM accuracy (pp)')
    ax.set_title('AE-SVM Advantage vs. Feature Dimensionality', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig10_dimensionality.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig10_dimensionality.png')


# =============================================================================
def fig11_uncertainty():
    df = pd.read_csv(DATA / 'qztna_uncertainty.csv')
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(df['threshold'], df['coverage'], color='#377eb8', marker='o', label='Coverage')
    ax1.set_xlabel('Uncertainty threshold $u$'); ax1.set_ylabel('Coverage', color='#377eb8')
    ax2 = ax1.twinx()
    ax2.plot(df['threshold'], df['accuracy'], color='#e41a1c', marker='s', label='Accuracy on retained set')
    ax2.set_ylabel('Accuracy', color='#e41a1c')
    ax1.set_title('Selective Prediction: Coverage / Accuracy vs. Threshold', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig11_uncertainty.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig11_uncertainty.png')


# =============================================================================
def fig12_perturbation():
    gen = NetworkFlowGenerator(seed=42)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mat = np.zeros((len(ATTACK_TYPES), gen.n_features))
    for i, atk in enumerate(ATTACK_TYPES):
        for j in gen.attack_dims[atk]:
            k = gen.attack_dims[atk].index(j)
            mat[i, j] = gen._attack_shifts[atk][k]
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(ATTACK_TYPES))); ax.set_yticklabels([ATTACK_LABELS[a] for a in ATTACK_TYPES])
    ax.set_xlabel('Feature index (header | flow | behavioral)')
    ax.axvline(gen.n_header - 0.5, color='k', lw=0.8, alpha=0.5)
    ax.axvline(gen.n_header + gen.n_flow - 0.5, color='k', lw=0.8, alpha=0.5)
    ax.set_title('Per-Attack Latent-Space Perturbation Magnitude', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Shift magnitude ($\\sigma$ units)')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig12_perturbation.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig12_perturbation.png')


# =============================================================================
def fig13_kernel_alignment():
    df = pd.read_csv(DATA / 'qztna_kernel_alignment.csv')
    piv = df.groupby(['kernel', 'attack'])['alignment'].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ATTACK_TYPES)); w = 0.25
    for i, kname in enumerate(['AE-SVM', 'RBF-SVM', 'Laplacian-SVM']):
        vals = [piv.loc[kname, a] if kname in piv.index else 0 for a in ATTACK_TYPES]
        ax.bar(x + (i - 1) * w, vals, w, label=kname, color=colors.get(kname))
    ax.set_xticks(x); ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_TYPES], rotation=15, ha='right')
    ax.set_ylabel('Kernel-target alignment')
    ax.set_title('Kernel-Target Alignment by Attack Type', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig13_kernel_alignment.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig13_kernel_alignment.png')


# =============================================================================
def fig14_feature_space():
    gen = NetworkFlowGenerator(seed=42)
    df = gen.generate_dataset(n_samples=1500, attack_ratio=0.35, random_state=1)
    cols = [c for c in df.columns if c.startswith('f')]
    X = df[cols].values
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    emb = TSNE(n_components=2, random_state=0, perplexity=30, init='pca').fit_transform(Xn)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, atk in enumerate(['normal'] + ATTACK_TYPES):
        mask = df['attack_type'] == atk
        label = 'Normal' if atk == 'normal' else ATTACK_LABELS[atk]
        color = '#999999' if atk == 'normal' else colors['attack'][i - 1]
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6, label=label, color=color)
    ax.set_title('t-SNE of Amplitude-Encoded Flow Features', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / 'fig14_feature_space.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig14_feature_space.png')


# =============================================================================
def fig15_landmarks():
    """Replaces the original (non-physical, SVM-has-no-epochs) 'training
    convergence' figure with a Nystrom landmark-count ablation: the actual
    accuracy/latency trade-off knob of the deployed AE-SVM."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from qztna.models.kernel_svm import NystromKernelSVM
    from sklearn.metrics import accuracy_score
    import time as _time

    gen = NetworkFlowGenerator(seed=42)
    df = gen.generate_dataset(n_samples=6000, attack_ratio=0.2, random_state=0)
    cols = [c for c in df.columns if c.startswith('f')]
    X, y = df[cols].values, df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    m_grid = [10, 25, 50, 100, 200, 400]
    accs, lats = [], []
    for m in m_grid:
        model = NystromKernelSVM(kernel='ae', random_state=0, landmark_frac=1.0, max_landmarks=m, min_landmarks=m)
        model.fit(Xtr, ytr)
        accs.append(accuracy_score(yte, model.predict(Xte)))
        t0 = _time.perf_counter()
        for _ in range(10):
            model.predict(Xte[:200])
        lats.append((_time.perf_counter() - t0) * 1000 / (10 * 200))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(m_grid, accs, marker='o', color=colors['AE-SVM'], linewidth=2)
    ax1.set_xlabel('Number of Nystrom landmarks $m$'); ax1.set_ylabel('Test accuracy')
    ax1.set_title('Accuracy vs. Landmark Count', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.25)
    ax2.plot(m_grid, lats, marker='s', color='#e41a1c', linewidth=2)
    ax2.set_xlabel('Number of Nystrom landmarks $m$'); ax2.set_ylabel('Latency per sample (ms)')
    ax2.set_title('Latency vs. Landmark Count', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig15_convergence.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    print('  Done: fig15_convergence.png (repurposed as landmark-count ablation)')


if __name__ == '__main__':
    fig1_framework()
    fig2_detection_performance()
    fig3_matched_n()
    fig4_certified_radii()
    fig5_sensitivity()
    fig6_zeroday()
    fig7_latency()
    fig8_confusion()
    fig9_roc()
    fig10_dimensionality()
    fig11_uncertainty()
    fig12_perturbation()
    fig13_kernel_alignment()
    fig14_feature_space()
    fig15_landmarks()
    print('All Q-ZTNA figures regenerated from real data.')
