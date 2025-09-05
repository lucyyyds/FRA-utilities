#!/usr/bin/env python3
"""
moku_eis_tool.py

Convert Moku FRA exports (two-input mode with Math Ch1/Ch2) into complex impedance,
plot Nyquist/Bode, and fit a Randles model (Ru + (Rct || C)).

- Reads a calibration YAML (kI A/V per current range), or accept --kI override.
- Handles I-monitor inversion automatically (or force with flags).
- Batch-process a folder of CSVs.

Author: ChatGPT
"""
import os, sys, math, argparse
from io import StringIO
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---- optional SciPy import for fitting ----
_HAS_SCIPY = True
try:
    from scipy.optimize import least_squares
except Exception:
    _HAS_SCIPY = False

# -----------------------------
# CSV parsing
# -----------------------------
def load_moku_csv(csv_path: str) -> pd.DataFrame:
    """Load a Moku FRA CSV (skips lines starting with '%')."""
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        data_lines = [ln for ln in f if not ln.strip().startswith("%") and ln.strip() != ""]
    names7 = ["Frequency_Hz",
              "Ch1_Mag_dB", "Ch1_Phase_deg",
              "Ch2_Mag_dB", "Ch2_Phase_deg",
              "Math_Mag_dB", "Math_Phase_deg"]
    try:
        df = pd.read_csv(StringIO("".join(data_lines)), header=None, names=names7)
    except Exception:
        df = pd.read_csv(StringIO("".join(data_lines)))
        if df.shape[1] >= 7:
            df.columns = names7 + [f"Extra_{i}" for i in range(df.shape[1]-7)]
        else:
            raise
    return df

def db_to_linear_mag(dB):  # 20*log10(V) -> linear magnitude
    return 10.0 ** (np.asarray(dB) / 20.0)

# -----------------------------
# Impedance computation
# -----------------------------
def compute_impedance(df: pd.DataFrame, kI: float, invert: str = "auto",
                      fmin=None, fmax=None):
    """
    Compute complex impedance Z from Moku FRA export using calibrated kI.
    invert: "auto" | "yes" | "no"
    fmin/fmax: optional frequency window (Hz)
    """
    freq = df["Frequency_Hz"].to_numpy(dtype=float)

    # Prefer Math ratio
    if not df["Math_Mag_dB"].isna().all() and not df["Math_Phase_deg"].isna().all():
        R_mag = db_to_linear_mag(df["Math_Mag_dB"].to_numpy(dtype=float))
        R_ph  = np.deg2rad(df["Math_Phase_deg"].to_numpy(dtype=float))
    else:  # Use Ch1/Ch2
        R_mag = db_to_linear_mag(df["Ch1_Mag_dB"] - df["Ch2_Mag_dB"])
        R_ph  = np.deg2rad(df["Ch1_Phase_deg"] - df["Ch2_Phase_deg"])

    R = R_mag * (np.cos(R_ph) + 1j*np.sin(R_ph))
    Z = (1.0 / kI) * R

    inverted = False
    if invert == "auto":
        mid = (freq > 1.0) & (freq < 1000.0)
        if np.nanmedian(np.real(Z[mid])) < 0:
            Z = -Z; inverted = True
    elif invert == "yes":
        Z = -Z; inverted = True
    elif invert == "no":
        inverted = False
    else:
        raise ValueError("invert must be 'auto', 'yes', or 'no'")

    if fmin is not None or fmax is not None:
        mask = np.ones_like(freq, dtype=bool)
        if fmin is not None: mask &= (freq >= float(fmin))
        if fmax is not None: mask &= (freq <= float(fmax))
        freq, Z = freq[mask], Z[mask]
    return freq, Z, inverted

# -----------------------------
# Randles model fit
# -----------------------------
def randles_Z(freq, Ru, Rct, C):
    w = 2*np.pi*freq
    Zpar = 1.0/(1.0/Rct + 1j*w*C)
    return Ru + Zpar

def fit_randles(freq, Z):
    mask = np.isfinite(np.real(Z)) & np.isfinite(np.imag(Z)) & np.isfinite(freq)
    f, Zm = freq[mask], Z[mask]
    Ru0 = max(1.0, np.nanmin(np.real(Zm)))
    Rct0 = max(1.0, np.nanmax(np.real(Zm)) - Ru0)
    idx_peak = np.nanargmax(-np.imag(Zm))
    f0 = f[idx_peak] if idx_peak is not None else 1.0
    C0 = 1.0/(2*np.pi*Rct0*f0) if (Rct0>0 and f0>0) else 1e-6
    if _HAS_SCIPY:
        def resid(p):
            Ru, Rct, C = p
            Zf = randles_Z(f, Ru, Rct, C)
            return np.r_[ (Zf.real - Zm.real), (Zf.imag - Zm.imag) ]
        from scipy.optimize import least_squares
        sol = least_squares(resid, x0=[Ru0,Rct0,C0], bounds=([0,0,0],[np.inf,np.inf,np.inf]), max_nfev=20000)
        Ru, Rct, C = sol.x; method = "scipy"
    else:
        Ru, Rct, C, method = Ru0, Rct0, C0, "init"
    return Ru, Rct, C, method

# -----------------------------
# I/O + plots
# -----------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_outputs(outdir, base, freq, Z, Ru, Rct, C, kI, inverted):
    ensure_dir(outdir)
    csv_path = os.path.join(outdir, f"{base}_impedance.csv")
    pd.DataFrame({"Frequency_Hz":freq,"Re_Z_ohm":np.real(Z),"Im_Z_ohm":np.imag(Z),"NegIm_Z_ohm":-np.imag(Z)}).to_csv(csv_path, index=False)
    txt_path = os.path.join(outdir, f"{base}_summary.txt")
    hf = float(np.nanmin(np.real(Z))); lf = float(np.nanmax(np.real(Z))); span = lf-hf
    idxp = int(np.nanargmax(-np.imag(Z))); f0_raw = float(freq[idxp]); negIm_peak = float(-np.imag(Z[idxp]))
    f0_fit = 1.0/(2*np.pi*Rct*C) if (Rct>0 and C>0) else float("nan")
    with open(txt_path,"w") as f:
        f.write(f"Input kI (A/V): {kI:.6g}\nInverted I-mon: {inverted}\n\n"
                "Nyquist read-offs (raw):\n"
                f"  HF intercept (Ru_raw) ~ {hf:.3f} Ω\n"
                f"  LF end (Ru+Rct_raw)  ~ {lf:.3f} Ω\n"
                f"  Span (Rct_raw)       ~ {span:.3f} Ω\n"
                f"  Peak(-Im) at f ≈ {f0_raw:.6g} Hz, height ≈ {negIm_peak:.3f} Ω\n\n"
                f"Randles fit:\n  Ru={Ru:.6g} Ω\n  Rct={Rct:.6g} Ω\n  C={C:.6g} F\n  Pred f0={f0_fit:.6g} Hz\n")
    # Plots
    plt.figure(); plt.plot(np.real(Z), -np.imag(Z), '.', ms=3); plt.xlabel("Re(Z) [Ω]"); plt.ylabel("-Im(Z) [Ω]"); plt.title("Nyquist"); plt.axis('equal'); plt.grid(True); plt.savefig(os.path.join(outdir, f"{base}_nyquist.png"), dpi=200, bbox_inches="tight")
    plt.figure(); plt.semilogx(freq, np.abs(Z)); plt.xlabel("Frequency [Hz]"); plt.ylabel("|Z| [Ω]"); plt.title("Bode magnitude"); plt.grid(True); plt.savefig(os.path.join(outdir, f"{base}_bode_mag.png"), dpi=200, bbox_inches="tight")
    plt.figure(); plt.semilogx(freq, np.rad2deg(np.angle(Z))); plt.xlabel("Frequency [Hz]"); plt.ylabel("Phase [deg]"); plt.title("Bode phase"); plt.grid(True); plt.savefig(os.path.join(outdir, f"{base}_bode_phase.png"), dpi=200, bbox_inches="tight")
    Zfit = randles_Z(freq, Ru, Rct, C)
    plt.figure(); plt.plot(np.real(Z), -np.imag(Z), '.', ms=3, label="Data"); plt.plot(np.real(Zfit), -np.imag(Zfit), '-', label=f"Fit: Ru={Ru:.2f}Ω, Rct={Rct:.2f}Ω, C={C:.2e}F"); plt.xlabel("Re(Z) [Ω]"); plt.ylabel("-Im(Z) [Ω]"); plt.title("Nyquist + Randles fit"); plt.legend(); plt.axis('equal'); plt.grid(True); plt.savefig(os.path.join(outdir, f"{base}_nyquist_fit.png"), dpi=200, bbox_inches="tight")
    return csv_path, txt_path

# -----------------------------
# Calibration loading
# -----------------------------
def load_calibration(yaml_path):
    import yaml
    with open(yaml_path,"r",encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y.get("ranges", {}), y

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Process Moku FRA CSV into impedance and fit a Randles model.")
    ap.add_argument("input", help="CSV file or directory of CSVs")
    ap.add_argument("--calib", default="calibration.yaml", help="YAML calibration file (kI per range)")
    ap.add_argument("--range", dest="current_range", help="Current range key, e.g. '10uA'")
    ap.add_argument("--kI", dest="kI_override", type=float, help="Override kI (A/V)")
    ap.add_argument("--invert", choices=["auto","yes","no"], default="auto", help="Invert I-monitor")
    ap.add_argument("--fmin", type=float, help="Lower frequency limit (Hz)")
    ap.add_argument("--fmax", type=float, help="Upper frequency limit (Hz)")
    ap.add_argument("--outdir", default="eis_out", help="Output directory")
    args = ap.parse_args()

    if args.kI_override is not None:
        kI = float(args.kI_override)
    else:
        ranges = {}
        if os.path.exists(args.calib):
            try:
                ranges, _ = load_calibration(args.calib)
            except Exception:
                ranges = {}
        if not args.current_range:
            ap.error("Provide --range or --kI.")
        if args.current_range not in ranges:
            ap.error(f"Range '{args.current_range}' not found in {args.calib}. Add it or use --kI.")
        kI = float(ranges[args.current_range])

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, n) for n in os.listdir(args.input) if n.lower().endswith(".csv")]
    else:
        files = [args.input]

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for fpath in sorted(files):
        try:
            df = load_moku_csv(fpath)
            freq, Z, inverted = compute_impedance(df, kI=kI, invert=args.invert, fmin=args.fmin, fmax=args.fmax)
            Ru, Rct, C, method = fit_randles(freq, Z)
            base = os.path.splitext(os.path.basename(fpath))[0]
            csv_out, txt_out = save_outputs(args.outdir, base, freq, Z, Ru, Rct, C, kI, inverted)
            rows.append({"file":fpath,"processed_csv":csv_out,"summary_txt":txt_out,"kI":kI,"inverted":inverted,"fit_method":method,"Ru":Ru,"Rct":Rct,"C":C})
            print(f"Processed: {fpath}")
        except Exception as e:
            print(f"ERROR processing {fpath}: {e}", file=sys.stderr)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir,"batch_summary.csv"), index=False)

if __name__ == "__main__":
    main()
