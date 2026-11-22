#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker_fit.py -- DESI DR2 BAO + Cosmic Chronometers: LambdaCDM vs tracker (rho_DE ~ H)
vs CPL + Om prior. Self-contained: DESI DR2 data auto-downloads on first run.

Usage (Colab: paste the whole file into one cell):
  python tracker_fit.py                       # real data
  python tracker_fit.py --selftest            # offline check on synthetic data
  python tracker_fit.py <desi_dir> --cc cc_moresco.csv

Requires: numpy, scipy (matplotlib optional, for fig_wz.png).
"""
import os, sys, math
import numpy as np
from scipy.optimize import minimize, least_squares

# ==== CC DATA (EDIT HERE): paste rows (z, H, sigma_H), H in km/s/Mpc ====
CC_DATA = [
    (0.0701, 69.0, 19.6), (0.090, 69.0, 12.0), (0.120, 68.6, 26.2),
    (0.170, 83.0, 8.0),   (0.179, 75.0, 4.0),   (0.199, 75.0, 5.0),
    (0.200, 72.9, 29.6),  (0.270, 77.0, 14.0),  (0.280, 88.8, 36.6),
    (0.352, 83.0, 14.0),  (0.380, 83.0, 13.5),  (0.400, 95.0, 17.0),
    (0.430, 77.0, 10.2),  (0.440, 87.1, 11.2),  (0.450, 92.8, 12.9),
    (0.470, 89.0, 49.6),  (0.480, 80.9, 9.0),   (0.593, 97.0, 62.0),
    (0.680, 104.0, 13.0), (0.730, 92.0, 8.0),   (0.781, 105.0, 12.0),
    (0.875, 125.0, 17.0), (0.880, 90.0, 40.0),  (0.900, 117.0, 23.0),
    (1.037, 154.0, 20.0), (1.300, 168.0, 17.0), (1.363, 160.0, 33.6),
    (1.430, 177.0, 18.0), (1.530, 140.0, 14.0), (1.750, 202.0, 40.0),
    (1.965, 186.5, 50.4),
]

c_kms = 299792.458
DESI_Z_FALLBACK = [0.295] + [z for z in (0.51, 0.71, 0.93, 1.33, 1.70, 2.33) for _ in range(2)]
DESI_KIND_FALLBACK = ["DV"] + ["DM", "DH"] * 6
DESI_BASES = [
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/",
    "https://cdn.jsdelivr.net/gh/CobayaSampler/bao_data@master/desi_bao_dr2/",
]
DESI_FILES = {"mean": "desi_gaussian_bao_ALL_GCcomb_mean.txt",
              "cov":  "desi_gaussian_bao_ALL_GCcomb_cov.txt"}

def fetch(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "python-tracker-fit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")

def ensure_desi(folder="desi_bao_dr2"):
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for key, fname in DESI_FILES.items():
        path = os.path.join(folder, fname)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            text, err = None, None
            for base in DESI_BASES:
                try:
                    text = fetch(base + fname); break
                except Exception as e:
                    err = e
            if text is None:
                print("ERROR: cannot download DESI files (%s)." % err)
                print("Manual download (put into %s/):" % folder)
                for base in DESI_BASES: print("   " + base + fname)
                sys.exit(1)
            open(path, "w", encoding="utf-8").write(text)
        paths[key] = path
    return paths

def _is_number(tok):
    try: float(tok); return True
    except ValueError: return False

def _split_tokens(line):
    return line.replace(",", " ").replace(";", " ").replace("\t", " ").split()

def _match_kind(words):
    s = " ".join(words).upper().replace("_", "").replace("/", "").replace("-", "")
    for key in ("DV", "DM", "DH"):
        if key in s: return key
    return None

def parse_mean_file(path, n_cov):
    """Flexible parser: (z, obs, val, sig) rows, (z, val) rows, (val, sig),
    (z, val, sig) or a bare column. Returns (zs, kinds, vals, sigmas) or None."""
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    rows, all_nums = [], []
    for ln in raw.splitlines():
        toks = [t for t in _split_tokens(ln) if t and not t.startswith("#")]
        if not toks: continue
        words = [t for t in toks if not _is_number(t)]
        nums = [float(t) for t in toks if _is_number(t)]
        if nums:
            rows.append((words, nums)); all_nums.extend(nums)
    if not rows: return None
    zs = kinds = vals = sigs = None
    if len(rows) == n_cov:
        words_first = [w for w, _ in rows]
        kinds = [_match_kind(w) for w in words_first]
        if all(len(n) >= 3 for _, n in rows):
            zs = [n[0] for _, n in rows]
            vals = [n[-2] for _, n in rows]
            sigs = [n[-1] for _, n in rows]
        elif all(len(n) == 2 for _, n in rows):
            a, b = [n[0] for _, n in rows], [n[1] for _, n in rows]
            amon = all(a[i + 1] >= a[i] - 1e-9 for i in range(len(a) - 1))
            bmon = all(b[i + 1] >= b[i] - 1e-9 for i in range(len(b) - 1))
            zlike = (all(0.0 < x <= 10.0 for x in a) and amon
                     and (len(set(round(x, 3) for x in a)) < len(a) or not bmon))
            if zlike:                       # DESI format: (z, value) per row
                zs, vals, sigs = a, np.array(b), None
                print("NOTE: mean file read as (z, value) rows; sigmas from covariance.")
            else:
                vals, sigs = np.array(a), np.array(b)
                print("NOTE: mean file read as (value, sigma) pairs.")
    if vals is None:
        if len(all_nums) == n_cov:
            vals, sigs, zs, kinds = all_nums, None, None, None
            print("NOTE: bare column of %d values; sigmas from cov diagonal." % n_cov)
        elif len(all_nums) == 2 * n_cov:
            a, b = all_nums[0::2], all_nums[1::2]
            amon = all(a[i + 1] >= a[i] - 1e-9 for i in range(len(a) - 1))
            bmon = all(b[i + 1] >= b[i] - 1e-9 for i in range(len(b) - 1))
            zlike = (all(0.0 < x <= 10.0 for x in a) and amon
                     and (len(set(round(x, 3) for x in a)) < len(a) or not bmon))
            if zlike:
                zs, vals, sigs = a, np.array(b), None
                print("NOTE: mean file read as (z, value) pairs.")
            else:
                vals, sigs = np.array(a), np.array(b)
                print("NOTE: mean file read as (value, sigma) pairs.")
        elif len(all_nums) == 3 * n_cov:
            zs, vals, sigs = all_nums[0::3], all_nums[1::3], all_nums[2::3]
            print("NOTE: mean file read as (z, value, sigma) triplets.")
    if vals is None or len(vals) != n_cov: return None
    if zs is not None and kinds is not None and any(k is None for k in kinds):
        kinds = None
    if zs is not None and vals is not None and len(zs) == len(vals):
        if max(abs(np.array(zs, float) - np.array(vals, float))) < 1e-6:
            print("ERROR: parsed values coincide with redshifts -- column shift; aborting.")
            return None
    if kinds is None or any(k is None for k in kinds):
        kinds2 = None
        if zs is not None:                      # group consecutive equal z
            kinds2, i, n = [], 0, len(zs)
            while i < n:
                j = i
                while j + 1 < n and abs(zs[j + 1] - zs[i]) <= 1e-6: j += 1
                m = j - i + 1
                kinds2.extend(["DV"] if m == 1 else ["DM", "DH"][:m])
                i = j + 1
        if kinds2 is None or any(k is None for k in kinds2):
            kinds2 = DESI_KIND_FALLBACK[:len(vals)]
            print("NOTE: observable kinds inferred from standard DESI ordering.")
        kinds = kinds2
    if zs is None or any(z is None for z in zs):
        print("NOTE: no redshifts in mean file; using standard DESI DR2 grid.")
        zs = DESI_Z_FALLBACK[:len(vals)]
    if sigs is None: sigs = [None] * len(vals)
    return list(map(float, zs)), kinds, np.array(vals), np.array(sigs)

def load_desi(folder="desi_bao_dr2"):
    paths = ensure_desi(folder)
    raw = open(paths["cov"], "r", encoding="utf-8", errors="replace").read()
    rows = []
    for ln in raw.splitlines():
        toks = [t for t in _split_tokens(ln) if t and not t.startswith("#")]
        nums = [float(t) for t in toks if _is_number(t)]
        if nums: rows.append(nums)
    cov = None
    if rows and len(rows) == len(rows[0]):
        cov = np.array(rows)
    elif rows:
        flat = [x for r in rows for x in r]
        n = int(round(math.sqrt(len(flat))))
        if n * n == len(flat): cov = np.array(flat).reshape(n, n)
    if cov is None:
        print("ERROR: covariance file not understood:"); print("\n".join(raw.splitlines()[:12]))
        sys.exit(1)
    parsed = parse_mean_file(paths["mean"], cov.shape[0])
    if parsed is None:
        print("ERROR: cannot parse mean file. First lines (send these for fixing):")
        print("\n".join(open(paths["mean"], encoding="utf-8", errors="replace").read().splitlines()[:12]))
        sys.exit(1)
    zs, kinds, vals, sigs = parsed
    if cov.shape[0] != len(vals):
        print("ERROR: shape mismatch: data %d vs cov %d." % (len(vals), cov.shape[0]))
        sys.exit(1)
    diag = np.sqrt(np.diag(cov))
    if sigs is not None and all(s is not None and s > 0 for s in sigs):
        ratio = diag / np.asarray(sigs, float)
        if np.max(np.abs(ratio - 1.0)) > 0.05:
            print("WARNING: file sigmas differ from sqrt(diag(cov)); using full covariance.")
    else:
        print("NOTE: no per-point sigmas; using covariance diagonal.")
        sigs = diag
    return list(map(float, zs)), kinds, np.asarray(vals, float), np.asarray(sigs, float), cov

def load_cc(cc_path=None, cc_array=None):
    rows = []
    if cc_path and os.path.exists(cc_path):
        for ln in open(cc_path, encoding="utf-8", errors="replace"):
            toks = [t for t in _split_tokens(ln) if t and not t.startswith("#")]
            nums = [float(t) for t in toks if _is_number(t)]
            if len(nums) >= 3: rows.append(nums[:3])
        if rows: return np.array(rows)
    if cc_array:
        rows = [r for r in cc_array if len(r) == 3]
        return np.array(rows) if rows else None
    return None

# ============================ cosmology ================================
def E_of(z, Om, w0=-1.0, wa=0.0, tracker=False):
    z = np.atleast_1d(np.asarray(z, float))
    a = 1.0 / (1.0 + z)
    if tracker:   # rho_DE ~ H: E^2 = Om a^-3 + (1-Om) E (flat, closed form)
        b = 1.0 - Om
        return 0.5 * (b + np.sqrt(b * b + 4.0 * Om * a ** (-3.0)))
    if w0 == -1.0 and wa == 0.0:
        return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))
    n = 3.0 * (1.0 + w0 + wa)
    de = (1.0 + z) ** n * np.exp(-3.0 * wa * z / (1.0 + z))
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om) * de)

_ZG = np.linspace(0.0, 3.0, 6001)

def DM_cH0(z, Om, w0=-1.0, wa=0.0, tracker=False):
    """D_M in units of c/H0."""
    invE = 1.0 / E_of(_ZG, Om, w0, wa, tracker)
    dm = np.empty_like(_ZG); dm[0] = 0.0
    dm[1:] = np.cumsum(0.5 * (invE[1:] + invE[:-1]) * np.diff(_ZG))
    scalar = np.ndim(z) == 0
    out = np.interp(np.atleast_1d(z), _ZG, dm)
    return float(out[0]) if scalar else out

def bao_pred(zs, kinds, H0, Om, rd, w0=-1.0, wa=0.0, tracker=False):
    out = []
    for z, kind in zip(zs, kinds):
        E = float(E_of(z, Om, w0, wa, tracker)[0])
        if kind == "DV":
            dm = DM_cH0(z, Om, w0, wa, tracker) * c_kms / H0
            dv = (dm ** 2 * z * (c_kms / H0 / E)) ** (1.0 / 3.0)
            out.append(dv / rd)
        elif kind == "DM":
            out.append(DM_cH0(z, Om, w0, wa, tracker) * c_kms / H0 / rd)
        else:
            out.append(c_kms / H0 / E / rd)
    return np.array(out)

def chi2_bao(params, data, model):
    zs, kinds, vals, invcov, rd_fix = data[0], data[1], data[2], data[5], data[6]
    if model == "lcdm":
        pred = bao_pred(zs, kinds, *params)
    elif model == "tracker":
        pred = bao_pred(zs, kinds, params[0], params[1], rd_fix, tracker=True)
    else:
        pred = bao_pred(zs, kinds, params[0], params[1], params[2], w0=params[3], wa=params[4])
    d = vals - pred
    return float(d @ invcov @ d)

def chi2_cc(params, cc, model):
    if cc is None: return 0.0
    z, Hobs, sig = cc[:, 0], cc[:, 1], cc[:, 2]
    if model == "lcdm":       E = E_of(z, params[1])
    elif model == "tracker":  E = E_of(z, params[1], tracker=True)
    else:                     E = E_of(z, params[1], w0=params[3], wa=params[4])
    return float(np.sum(((params[0] * E - Hobs) / sig) ** 2))

def chi2_total(params, data, cc, model):
    prior = ((params[1] - 0.31) / 0.02) ** 2 if model == "cpl" else 0.0
    return chi2_bao(params, data, model) + chi2_cc(params, cc, model) + prior

def penalty(params, model):
    if model == "cpl":     H0, Om, rd, w0, wa = params
    elif model == "tracker": H0, Om, rd = params[0], params[1], 147.0
    else:                  H0, Om, rd = params[0], params[1], params[2]
    bad = 0.0
    if not (40.0 < H0 < 100.0): bad += (H0 - 70.0) ** 4
    if not (0.03 < Om < 0.60):  bad += (Om - 0.30) ** 4
    if model != "tracker" and not (50.0 < rd < 250.0): bad += (rd - 147.0) ** 4
    if model == "cpl":
        if not (-3.0 < w0 < 1.0): bad += (w0 + 1.0) ** 4
        if not (-4.0 < wa < 4.0): bad += wa ** 4
    return bad

def fit_model(model, data, cc, starts):
    best = None
    for x0 in starts:
        try:
            res = minimize(lambda p: chi2_total(list(p), data, cc, model) + penalty(p, model),
                           np.array(x0, float), method="Nelder-Mead",
                           options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True})
            if best is None or res.fun < best.fun: best = res
        except Exception:
            pass
    res = minimize(lambda p: chi2_total(list(p), data, cc, model) + penalty(p, model),
                   np.array(best.x, float), method="Nelder-Mead",
                   options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True})
    if res.fun < best.fun: best = res
    return np.array(best.x, float), float(best.fun)

def hessian_errors(x0, data, cc, model):
    n = len(x0); h = np.array([max(1e-3 * abs(v), 1e-4) for v in x0])
    H = np.zeros((n, n)); f0 = chi2_total(list(x0), data, cc, model)
    for i in range(n):
        for j in range(i, n):
            xi = np.array(x0, float); xi[i] += h[i]
            xj = np.array(x0, float); xj[j] += h[j]
            xij = xi.copy(); xij[j] += h[j]
            H[i, j] = H[j, i] = (chi2_total(list(xij), data, cc, model)
                                 - chi2_total(list(xi), data, cc, model)
                                 - chi2_total(list(xj), data, cc, model) + f0) / (h[i] * h[j])
    cov = np.linalg.pinv(H)
    return np.sqrt(np.abs(np.diag(cov))), cov

def tracker_effective_cpl(Om, zmax=2.33):
    """Fit (w0, wa) of CPL to the tracker density ratio rho/rho0 = E(z)."""
    zt = np.linspace(0.02, zmax, 40)
    lnE = np.log(E_of(zt, Om, tracker=True))
    def resid(p):
        w0, wa = p
        return 3.0 * (1.0 + w0 + wa) * np.log1p(zt) - 3.0 * wa * zt / (1.0 + zt) - lnE
    sol = least_squares(resid, [-0.8, 0.3])
    return float(sol.x[0]), float(sol.x[1])

def report_fit(name, model, params, data, cc, rd_fix=None):
    chi2 = chi2_total(list(params), data, cc, model)
    bao, ccc = chi2_bao(params, data, model), chi2_cc(params, cc, model)
    n_data = len(data[2]) + (0 if cc is None else len(cc))
    k = 5 if model == "cpl" else (3 if model == "lcdm" else 2)
    if model == "cpl":
        H0, Om, rd, w0, wa = params; extra = "w0=%.3f wa=%.3f " % (w0, wa)
    elif model == "tracker":
        H0, Om, rd = params[0], params[1], rd_fix; extra = "rd=fix "
    else:
        H0, Om, rd = params; extra = ""
    print("  %-9s H0=%6.2f Om=%.4f rd=%6.1f %s| chi2=%7.3f (BAO %6.3f, CC %6.3f) chi2/dof=%.3f"
          % (name, H0, Om, rd, extra, chi2, bao, ccc, chi2 / max(n_data - k, 1)))
    return chi2

# ============================ selftest =================================
def run_selftest():
    print("== SELFTEST: synthetic LambdaCDM data (offline, no downloads) ==")
    H0t, Omt, rdt = 70.0, 0.30, 147.1
    zs, kinds = DESI_Z_FALLBACK, DESI_KIND_FALLBACK
    vals = bao_pred(zs, kinds, H0t, Omt, rdt)
    sigs = np.where(np.array(kinds) == "DV", 0.010, 0.008) * vals
    rng = np.random.RandomState(11)
    vals = vals * (1.0 + rng.normal(0.0, 0.004, size=vals.shape))
    cov = np.diag(sigs ** 2)
    data = [zs, kinds, vals, sigs, cov, np.linalg.inv(cov), None]
    zcc = np.linspace(0.1, 1.9, 20)
    Hcc = H0t * E_of(zcc, Omt) * (1.0 + rng.normal(0.0, 0.01, size=zcc.shape))
    cc = np.column_stack([zcc, Hcc, 0.04 * Hcc])
    print("Synthetic BAO: %d points, CC: %d points" % (len(vals), len(cc)))
    r1, _ = fit_model("lcdm", data, cc, [[70.0, 0.30, 147.1], [68.0, 0.28, 148.0]])
    report_fit("LambdaCDM", "lcdm", r1, data, cc)
    data_tr = list(data); data_tr[6] = r1[2]
    r2, _ = fit_model("tracker", data_tr, cc, [[70.0, 0.30], [66.0, 0.26]])
    report_fit("Tracker", "tracker", r2, data_tr, cc, rd_fix=r1[2])
    w0t, wat = tracker_effective_cpl(r2[1])
    print("  Tracker effective CPL: w0=%.3f wa=%.3f" % (w0t, wat))
    r3, _ = fit_model("cpl", data, cc, [[70.0, 0.30, 147.1, -1.0, 0.0],
                                        [66.0, 0.26, 147.0, -0.9, 0.2]])
    err, _ = hessian_errors(r3, data, cc, "cpl")
    report_fit("CPL+prior", "cpl", r3, data, cc)
    print("  CPL errors: w0=%.2f wa=%.2f" % (err[3], err[4]))
    print("== SELFTEST DONE ==")

# ================================ main =================================
def main():
    argv = sys.argv[1:]
    # Colab/Jupyter injects "-f <kernel.json>"; strip flag-value pairs,
    # stray flags and .json paths so only real arguments survive
    cleaned, i = [], 0
    while i < len(argv):
        if argv[i] == "-f":
            i += 2; continue
        if argv[i].startswith("-"):
            i += 1; continue
        if argv[i].endswith(".json"):
            i += 1; continue
        cleaned.append(argv[i]); i += 1
    argv = cleaned
    cc_path = None
    if "--cc" in argv:
        j = argv.index("--cc")
        cc_path = argv[j + 1] if j + 1 < len(argv) else None
        argv = argv[:j] + argv[j + 2:]
    if "--selftest" in argv:
        run_selftest(); return
    folder = argv[0] if argv else "desi_bao_dr2"
    if os.path.exists(folder) and not os.path.isdir(folder):
        folder = "desi_bao_dr2"          # guard: never treat a file as a folder
    print("== Loading DESI DR2 BAO (auto-download, cached in %s/) ==" % folder)
    zs, kinds, vals, sigs, cov = load_desi(folder)
    cc = load_cc(cc_path, CC_DATA)
    if cc is None:
        print("NOTE: no CC data (cc_moresco.csv or CC_DATA empty) -> BAO-only.")
    data = [zs, kinds, vals, sigs, cov, np.linalg.inv(cov), None]
    print("Parsed %d BAO points:" % len(vals))
    for z, k, v, s in zip(zs, kinds, vals, sigs):
        print("   z=%-6.3f %-2s obs=%10.5f sigma=%9.5f" % (z, k, v, s))
    if cc is not None:
        print("Loaded %d cosmic chronometers from %s." % (len(cc), cc_path or "CC_DATA"))
    r1, _ = fit_model("lcdm", data, cc, [[70.0, 0.30, 147.0], [68.0, 0.27, 148.0],
                                         [72.0, 0.33, 146.0]])
    print("\nFits:")
    chi2_l = report_fit("LambdaCDM", "lcdm", r1, data, cc)
    data_tr = list(data); data_tr[6] = r1[2]
    r2, _ = fit_model("tracker", data_tr, cc, [[70.0, 0.30], [66.0, 0.26]])
    chi2_t = report_fit("Tracker", "tracker", r2, data_tr, cc, rd_fix=r1[2])
    w0t, wat = tracker_effective_cpl(r2[1])
    print("  Tracker effective CPL: w0=%.3f wa=%.3f" % (w0t, wat))
    r3, _ = fit_model("cpl", data, cc, [[70.0, 0.30, 147.0, -1.0, 0.0],
                                        [66.0, 0.26, 147.0, -0.9, 0.2],
                                        [73.0, 0.24, 148.0, -0.8, -0.5]])
    err, _ = hessian_errors(r3, data, cc, "cpl")
    chi2_c = report_fit("CPL+prior", "cpl", r3, data, cc)
    print("  CPL errors: w0=%.2f wa=%.2f" % (err[3], err[4]))
    n_data = len(vals) + (0 if cc is None else len(cc))
    print("\nModel comparison:")
    for name, x2, k in (("LambdaCDM", chi2_l, 3), ("Tracker  ", chi2_t, 2),
                        ("CPL+prior", chi2_c, 5)):
        print("  %-9s chi2=%8.3f AIC=%8.3f BIC=%8.3f"
              % (name, x2, x2 + 2 * k, x2 + k * np.log(n_data)))
    print("  Delta chi2 (Tracker - LambdaCDM) = %+.3f" % (chi2_t - chi2_l))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zz = np.linspace(0.0, 2.4, 200)
        EE = E_of(zz, r2[1], tracker=True)
        wt = -1.0 + (1.0 / 3.0) * (-(1.0 + zz) * np.gradient(np.log(EE), zz))
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.axhline(-1.0, color="gray", ls="--", lw=1, label=r"$\Lambda$CDM")
        ax.plot(zz, wt, color="tab:blue", lw=2, label="tracker (rho~H)")
        ax.plot(zz, r3[3] + r3[4] * zz / (1.0 + zz), color="tab:orange", lw=2,
                label="CPL best fit")
        ax.set_xlabel("z"); ax.set_ylabel("w(z)"); ax.set_ylim(-2.0, 0.0)
        ax.legend(); fig.tight_layout(); fig.savefig("fig_wz.png", dpi=150)
        print("\nSaved fig_wz.png")
    except Exception as e:
        print("\n(plot skipped: %s)" % e)

if __name__ == "__main__":
    main()
