[README.md](https://github.com/user-attachments/files/35227008/README.md)
# M4 Cosmology Analysis

Reproducible numerical pipeline for the paper
**"Three Phenomena from One Foam: Dark Energy, Dark Matter, and Baryogenesis
as Coarse-Graining Limits of Quantum Foam on M4"** (E. A. Tarasenko).

## What this reproduces

Confrontation of three dark-energy realizations with the official
**DESI DR2 BAO likelihood** (13 observables, full 13x13 covariance) plus
**31 cosmic chronometers**, fitting (H0, Om, rd):

| Model | chi2 (BAO+CC) | Notes |
|---|---|---|
| LambdaCDM | 24.93 | H0=69.1, Om=0.298, rd=146.8 |
| Tracker (rho_DE ~ H) | 29.85 | zero free DE parameters; Delta chi2 = +4.9 |
| CPL (w0, wa) + Om prior | 23.20 | w0=-0.88+-0.11, wa=+0.16+-0.32 |

Key results:
- Tracker effective equation of state: **w0_eff = -0.79**, **wa_eff = +0.35** --
  opposite in sign to the DESI+CMB+SN preference (wa = -0.86 +- 0.22).
- On BAO+CC data alone the preference for evolving dark energy is
  statistically insignificant (Delta chi2 = -1.7 for two CPL parameters;
  Delta BIC = +5.8 vs LambdaCDM).
- The discriminating observable for the rho ~ H response class is the
  **sign of wa** on DESI DR3 / Euclid.

## Repository contents

```
tracker_fit.py          # full pipeline; DESI DR2 BAO data auto-downloads on first run
cc_moresco.csv          # 31 cosmic chronometers (Magana et al. 2018, arXiv:1802.01505)
```

The DESI DR2 BAO files (mean vector and 13x13 covariance of the
`desi_gaussian_bao_ALL_GCcomb` likelihood) are fetched automatically from the
public Cobaya data package (`CobayaSampler/bao_data`, folder `desi_bao_dr2`)
on the first run and cached in `./desi_bao_dr2/`. No manual download is
needed; internet access is required only once.

## How to run

```
python tracker_fit.py                       # full pipeline (auto-download)
python tracker_fit.py --selftest            # offline check on synthetic data
python tracker_fit.py desi_bao_dr2 --cc cc_moresco.csv
```

Requires Python 3.9+ with `numpy` and `scipy` (matplotlib optional, for
`fig_wz.png`). The script prints all fit results, the AIC/BIC comparison,
the chi2 triangle, 1-sigma errors of the CPL fit (numerical Hessian), and
writes `fig_wz.png`.

## Citation

If you use this pipeline, please cite the companion preprint (Zenodo DOI,
to be added) and the DESI DR2 BAO release. The cosmic chronometer
compilation follows Magana et al. 2018 (arXiv:1802.01505).

## License

MIT.
