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
| LambdaCDM | 25.71 | H0=68.8, Om=0.298, rd=147.6 (BAO 10.27 + CC 15.44) |
| Tracker (rho_DE ~ H) | 31.04 | zero free DE parameters; Delta chi2 = +5.3 |
| CPL (w0, wa) + Om prior | 23.81 | w0=-0.78+-0.13, wa=-0.61+-0.47 |

Key results:
- Tracker effective equation of state: **w0_eff = -0.80**, **wa_eff = +0.35** —
  opposite in sign to the DESI+CMB+SN preference (wa = -0.86 +- 0.22).
- On BAO+CC data alone the preference for evolving dark energy is
  statistically insignificant (Delta chi2 = -1.9 for two CPL parameters;
  Delta BIC = +5.7 vs LambdaCDM).
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

If you use this pipeline, please cite the companion preprint, https://doi.org/10.5281/zenodo.24434904, and the DESI DR2 BAO release. 
The cosmic chronometer compilation follows Magana et al. 2018 (arXiv:1802.01505).

## License

MIT.
