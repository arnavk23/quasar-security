`qmi_kernels.csv` is written by `scripts/qmi_experiments.py` and feeds `qmi/qmi.tex`.

`qmi_kernels_narrow_grid.csv` is the first run, kept for provenance. It used an RBF width grid of
{0.1, 0.3, 1, 3}/q and no Laplacian kernel. Its "QKD" rows re-standardized the Mahalanobis column,
which is the setting now called "QKD, centred Mahal.". The paper cites only its CIC-IDS2017 RBF and
ZZ held-out AUCs (0.672 and 0.874), to show how a narrow classical grid inflated the ZZ result.
