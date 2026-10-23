import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import beta as beta_fn, betainc

def beta_pdf(v, a, b):
    B = beta_fn(a, b); v = np.asarray(v); out = np.zeros_like(v, dtype=float)
    m = (v >= 0) & (v <= 1); out[m] = (v[m]**(a-1) * (1-v[m])**(b-1)) / B; return out

def beta_cdf(v, a, b): return betainc(a, b, np.clip(v,0,1))

def bidding_function_beta(n, rho, a, b, grid_size=2000):
    vg = np.linspace(0.0, 1.0, grid_size); bg = np.zeros_like(vg)
    for k in range(1, grid_size):
        v = vg[k-1]; dv = vg[k] - v; F = float(beta_cdf(v, a, b)); f = float(beta_pdf(v, a, b))
        db = 0.0 if F <= 1e-12 else ((n-1)/(1.0 - rho)) * (f / F) * (v - bg[k-1])
        bg[k] = max(0.0, bg[k-1] + dv * db)
    return vg, bg

def simulate_revenues(n, rho, a, b, N=30000, v_grid=None, b_grid=None, seed=7):
    rng = np.random.default_rng(seed)
    V = rng.beta(a, b, size=(N, n)); Vs = np.sort(V, axis=1)
    v_max = Vs[:, -1]; v_second = Vs[:, -2]
    rev_spa = float(np.mean(v_second))
    if v_grid is None or b_grid is None: v_grid, b_grid = bidding_function_beta(n, rho, a, b)
    rev_fpa = float(np.mean(np.interp(v_max, v_grid, b_grid)))
    return rev_fpa, rev_spa

def calibrate_rho_from_ratio(n, a, b, target_ratio, rho_grid=np.linspace(0.0, 0.9, 46)):
    rows = []
    for rho in rho_grid:
        vg, bg = bidding_function_beta(n, rho, a, b, grid_size=1200)
        r_fpa, r_spa = simulate_revenues(n, rho, a, b, N=20000, v_grid=vg, b_grid=bg, seed=123)
        ratio = r_fpa / r_spa if r_spa > 0 else np.nan
        rows.append({"rho": rho, "R_FPA": r_fpa, "R_SPA": r_spa, "ratio": ratio, "sq_err": (ratio - target_ratio)**2})
    tab = pd.DataFrame(rows).sort_values("sq_err")
    return float(tab.iloc[0]["rho"]), tab

def plot_revenue_vs_rho(n, a, b, rho_vals, seed=99):
    R_FPA, R_SPA = [], []
    for rho in rho_vals:
        vg, bg = bidding_function_beta(n, rho, a, b, grid_size=1000)
        r_fpa, r_spa = simulate_revenues(n, rho, a, b, N=20000, v_grid=vg, b_grid=bg, seed=seed)
        R_FPA.append(r_fpa); R_SPA.append(r_spa)
    plt.figure()
    plt.plot(rho_vals, R_FPA, label="First-Price Revenue")
    plt.plot(rho_vals, R_SPA, label="Second-Price Revenue")
    plt.xlabel("Risk aversion ρ"); plt.ylabel("Expected revenue")
    plt.title(f"Expected Revenue vs ρ (Beta({a},{b}), n={n})")
    plt.legend(); plt.tight_layout(); plt.savefig("figs_revenue_vs_rho.png", dpi=160)

def plot_gap_multi(n_list, a, b, rho_vals):
    plt.figure()
    for n in n_list:
        R_FPA, R_SPA = [], []
        for rho in rho_vals:
            vg, bg = bidding_function_beta(n, rho, a, b, grid_size=1000)
            r_fpa, r_spa = simulate_revenues(n, rho, a, b, N=20000, v_grid=vg, b_grid=bg, seed=100+n)
            R_FPA.append(r_fpa); R_SPA.append(r_spa)
        gap = np.array(R_FPA) - np.array(R_SPA)
        plt.plot(rho_vals, gap, label=f"n={n}")
    plt.xlabel("Risk aversion ρ"); plt.ylabel("Revenue advantage (FPA - SPA)")
    plt.title(f"FPA Advantage vs ρ (Beta({a},{b}))")
    plt.legend(); plt.tight_layout(); plt.savefig("figs_gap_vs_rho_multi_n.png", dpi=160)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "calibrate", "both"], default="both")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--b", type=float, default=3.0)
    p.add_argument("--target_ratio", type=float, default=1.08)
    args = p.parse_args()

    rho_vals = np.linspace(0.0, 0.9, 31)

    if args.mode in ["demo", "both"]:
        plot_revenue_vs_rho(args.n, args.a, args.b, rho_vals)
        plot_gap_multi([3,5,8], args.a, args.b, rho_vals)
        print("Saved: figs_revenue_vs_rho.png, figs_gap_vs_rho_multi_n.png")

    if args.mode in ["calibrate", "both"]:
        rho_hat, tab = calibrate_rho_from_ratio(args.n, args.a, args.b, args.target_ratio)
        tab.to_csv("calibration_grid_results.csv", index=False)
        print(f"rho* (calibrated) ≈ {rho_hat:.3f}")
        print("Saved: calibration_grid_results.csv")