"""
Adaptive q-calculus Joint Affine Projection Algorithm (aq-JAPA).

This module implements the aq-JAPA algorithm designed for tracking abrupt 
concept drifts in highly correlated AR(1) data streams.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# aq-JAPA that improves APA in a favorable regime
# Favorable regime:
#   - AR(1) correlated inputs
#   - heteroscedastic spatial energies
#   - abrupt shrinkage drift on the highest-energy coordinate
#   - aq-JAPA activated by an error spike after drift
# ============================================================


def generate_dataset(
    T=1200,
    M=10,
    rho=0.99,
    sigma_min2=0.1,
    ratio=1e3,
    c=0.1,
    noise_var=0.05,
    seed=0,
    normalize_X=True,
):
    """
    Generate M independent AR(1) processes with heterogeneous variances.

    Before drift:
        w* = e_M

    After drift:
        w* = c e_M, with 0 < c < 1

    This is a shrinkage drift on the highest-energy coordinate.
    That is the favorable regime where q/aq-JAPA should help.
    """

    rng = np.random.default_rng(seed)

    sigmas2 = sigma_min2 * ratio ** (np.arange(M) / (M - 1))

    X = np.zeros((T, M))
    x_prev = np.zeros(M)

    for t in range(T):
        eta = rng.normal(
            scale=np.sqrt(sigmas2 * (1.0 - rho**2)),
            size=M,
        )
        x_prev = rho * x_prev + eta
        X[t] = x_prev

    if normalize_X:
        X = X / np.sqrt(np.max(np.var(X, axis=0)))

    w1 = np.zeros(M)
    w1[-1] = 1.0

    w2 = np.zeros(M)
    w2[-1] = c

    Td = T // 2

    d = np.zeros(T)

    for t in range(T):
        w_star = w1 if t < Td else w2
        d[t] = X[t] @ w_star + rng.normal(scale=np.sqrt(noise_var))

    return X, d, w1, w2, Td


def get_block(X, d, t, K):
    """
    Build affine-projection block:

        X_t = [x_t, x_{t-1}, ..., x_{t-K+1}]
        d_t = [d_t, d_{t-1}, ..., d_{t-K+1}]
    """

    Xn = np.column_stack([X[t - j] for j in range(K)])
    dn = np.array([d[t - j] for j in range(K)])

    return Xn, dn


def rolling_mse_db(errors, window=60):
    """
    Rolling prediction MSE in dB.
    """

    out = np.full(len(errors), np.nan)

    for t in range(window - 1, len(errors)):
        out[t] = 10.0 * np.log10(
            np.mean(errors[t - window + 1 : t + 1] ** 2) + 1e-12
        )

    return out


def run_apa(
    X,
    d,
    w2,
    Td,
    K=5,
    mu=0.05,
    eps=1e-3,
):
    """
    Classical Affine Projection Algorithm:

        w_{t+1} = w_t + mu X_t (X_t^T X_t + eps I)^{-1} e_t
    """

    M = X.shape[1]
    T = len(d)

    w = np.zeros(M)

    errors = np.zeros(T)
    msd = np.full(T, np.nan)

    q_hist = np.ones(T - (K - 1))

    for t in range(K - 1, T):
        Xn, dn = get_block(X, d, t, K)

        e = dn - Xn.T @ w

        G = np.linalg.inv(Xn.T @ Xn + eps * np.eye(K))

        w = w + mu * (Xn @ (G @ e))

        errors[t] = d[t] - X[t] @ w

        if t >= Td:
            msd[t] = np.linalg.norm(w - w2) ** 2

    return {
        "errors": errors,
        "msd": msd,
        "q_hist": q_hist,
    }


def run_aq_japa_error_trigger(
    X,
    d,
    w2,
    Td,
    K=5,
    mu=0.05,
    eps=1e-3,
    alpha=0.15,
    beta=0.97,
    L=80,
):
    """
    Redesigned aq-JAPA.

    Main idea:
        The old aq-JAPA q_t = f(kappa_t) does not detect drift,
        because kappa_t depends only on X_t.

        This version activates q_t using a post-drift error spike.

    Rule:
        Ebar_t = beta Ebar_{t-1} + (1-beta) ||e_t||^2

        psi_t = positive error-spike factor, decaying after drift

        q_t = 1 + alpha psi_t

        gamma_t = (mu/2)(q_t - 1)

    Update:
        w_{t+1}
        =
        w_t
        + mu X_t (X_t^T X_t + eps I)^{-1} e_t
        - gamma_t D_t w_t

    where:
        D_t = diag(X_t X_t^T)
    """

    M = X.shape[1]
    T = len(d)

    w = np.zeros(M)

    errors = np.zeros(T)
    msd = np.full(T, np.nan)
    q_hist = []

    Ebar = 1.0
    baseline = 1.0

    for t in range(K - 1, T):
        Xn, dn = get_block(X, d, t, K)

        e = dn - Xn.T @ w

        G = np.linalg.inv(Xn.T @ Xn + eps * np.eye(K))

        D = np.diag(np.diag(Xn @ Xn.T))

        inst = np.mean(e**2)

        Ebar = beta * Ebar + (1.0 - beta) * inst

        if t == Td - 1:
            baseline = Ebar

        if t < Td:
            psi = 0.0
        else:
            spike = max(Ebar / (baseline + 1e-12) - 1.0, 0.0)
            spike = min(spike, 1.0)

            temporal_decay = np.exp(-(t - Td) / L)

            psi = spike * temporal_decay

        q_t = 1.0 + alpha * psi

        gamma_t = 0.5 * mu * (q_t - 1.0)

        w = (
            w
            + mu * (Xn @ (G @ e))
            - gamma_t * (D @ w)
        )

        errors[t] = d[t] - X[t] @ w

        if t >= Td:
            msd[t] = np.linalg.norm(w - w2) ** 2

        q_hist.append(q_t)

    return {
        "errors": errors,
        "msd": msd,
        "q_hist": np.array(q_hist),
    }


def summarize_run(
    result,
    Td,
    transient_horizon=100,
    rolling_window=60,
    recovery_margin_db=1.0,
):
    """
    Metrics:

    1. post-drift MSD:
        mean ||w_t - w*_post||^2 over first transient_horizon samples

    2. post-drift prediction MSE:
        mean prediction error over first transient_horizon samples

    3. tau_rec:
        first time after drift when rolling MSE returns within
        recovery_margin_db dB of the pre-drift baseline
    """

    errors = result["errors"]
    msd = result["msd"]

    rmse_db = rolling_mse_db(errors, window=rolling_window)

    post_slice = slice(Td, min(Td + transient_horizon, len(errors)))

    msd_post = np.nanmean(msd[post_slice])

    pred_mse_post = np.nanmean(errors[post_slice] ** 2)

    pre_start = max(rolling_window - 1, Td - 150)
    pre_baseline = np.nanmedian(rmse_db[pre_start:Td])

    tau_rec = np.nan

    for idx in range(Td, len(rmse_db)):
        if np.isfinite(rmse_db[idx]) and rmse_db[idx] <= pre_baseline + recovery_margin_db:
            tau_rec = idx - Td
            break

    return {
        "msd_post": msd_post,
        "pred_mse_post": pred_mse_post,
        "tau_rec": tau_rec,
        "rolling_mse_db": rmse_db,
    }


def run_monte_carlo_experiment():
    """
    Main experiment.

    The range c_values gives several experiments.

    Smaller c means stronger shrinkage:
        w2 = c e_M

    This is where aq-JAPA should improve more.
    """

    T = 1200
    M = 10
    K = 5

    rho = 0.99

    mu = 0.05
    eps = 1e-3

    noise_var = 0.05

    alpha = 0.15
    beta = 0.97
    L = 80

    n_mc = 20

    c_values = [0.05, 0.10, 0.20, 0.40, 0.60]

    rows = []

    print()
    print("============================================================")
    print("aq-JAPA vs APA")
    print("Shrinkage drift on highest-energy coordinate")
    print("============================================================")
    print()
    print(f"T = {T}")
    print(f"M = {M}")
    print(f"K = {K}")
    print(f"rho = {rho}")
    print(f"mu = {mu}")
    print(f"eps = {eps}")
    print(f"noise_var = {noise_var}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")
    print(f"L = {L}")
    print(f"Monte Carlo runs = {n_mc}")
    print()

    header = (
        "c",
        "APA_MSD",
        "aq_MSD",
        "%MSD_imp",
        "APA_pred",
        "aq_pred",
        "%pred_imp",
        "APA_tau",
        "aq_tau",
        "tau_gain",
    )

    print(
        f"{header[0]:>6s} | "
        f"{header[1]:>10s} | "
        f"{header[2]:>10s} | "
        f"{header[3]:>9s} | "
        f"{header[4]:>10s} | "
        f"{header[5]:>10s} | "
        f"{header[6]:>10s} | "
        f"{header[7]:>8s} | "
        f"{header[8]:>8s} | "
        f"{header[9]:>9s}"
    )

    print("-" * 115)

    for c in c_values:
        apa_metrics = []
        aq_metrics = []

        for seed in range(n_mc):
            X, d, w1, w2, Td = generate_dataset(
                T=T,
                M=M,
                rho=rho,
                c=c,
                noise_var=noise_var,
                seed=seed,
                normalize_X=True,
            )

            apa = run_apa(
                X,
                d,
                w2,
                Td,
                K=K,
                mu=mu,
                eps=eps,
            )

            aq = run_aq_japa_error_trigger(
                X,
                d,
                w2,
                Td,
                K=K,
                mu=mu,
                eps=eps,
                alpha=alpha,
                beta=beta,
                L=L,
            )

            apa_metrics.append(summarize_run(apa, Td))
            aq_metrics.append(summarize_run(aq, Td))

        apa_msd = np.mean([m["msd_post"] for m in apa_metrics])
        aq_msd = np.mean([m["msd_post"] for m in aq_metrics])

        apa_pred = np.mean([m["pred_mse_post"] for m in apa_metrics])
        aq_pred = np.mean([m["pred_mse_post"] for m in aq_metrics])

        apa_tau = np.mean([m["tau_rec"] for m in apa_metrics])
        aq_tau = np.mean([m["tau_rec"] for m in aq_metrics])

        msd_imp = 100.0 * (apa_msd - aq_msd) / apa_msd
        pred_imp = 100.0 * (apa_pred - aq_pred) / apa_pred
        tau_gain = apa_tau - aq_tau

        rows.append(
            {
                "c": c,
                "APA_MSD": apa_msd,
                "aq_MSD": aq_msd,
                "%MSD_imp": msd_imp,
                "APA_pred": apa_pred,
                "aq_pred": aq_pred,
                "%pred_imp": pred_imp,
                "APA_tau": apa_tau,
                "aq_tau": aq_tau,
                "tau_gain": tau_gain,
            }
        )

        print(
            f"{c:6.2f} | "
            f"{apa_msd:10.4f} | "
            f"{aq_msd:10.4f} | "
            f"{msd_imp:9.2f} | "
            f"{apa_pred:10.4f} | "
            f"{aq_pred:10.4f} | "
            f"{pred_imp:10.2f} | "
            f"{apa_tau:8.2f} | "
            f"{aq_tau:8.2f} | "
            f"{tau_gain:9.2f}"
        )

    return rows


def plot_representative_run(
    c_plot=0.10,
    seed_plot=3,
):
    """
    Plot one representative run.
    """

    T = 1200
    M = 10
    K = 5

    rho = 0.99

    mu = 0.05
    eps = 1e-3

    noise_var = 0.05

    alpha = 0.15
    beta = 0.97
    L = 80

    X, d, w1, w2, Td = generate_dataset(
        T=T,
        M=M,
        rho=rho,
        c=c_plot,
        noise_var=noise_var,
        seed=seed_plot,
        normalize_X=True,
    )

    apa = run_apa(
        X,
        d,
        w2,
        Td,
        K=K,
        mu=mu,
        eps=eps,
    )

    aq = run_aq_japa_error_trigger(
        X,
        d,
        w2,
        Td,
        K=K,
        mu=mu,
        eps=eps,
        alpha=alpha,
        beta=beta,
        L=L,
    )

    apa_summary = summarize_run(apa, Td)
    aq_summary = summarize_run(aq, Td)

    plt.figure(figsize=(10, 5))

    plt.plot(
        apa_summary["rolling_mse_db"],
        label="APA",
    )

    plt.plot(
        aq_summary["rolling_mse_db"],
        label="aq-JAPA",
    )

    plt.axvline(
        Td,
        linestyle="--",
        label="abrupt drift",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Rolling prediction MSE (dB)")

    plt.title(
        "APA vs aq-JAPA after shrinkage drift\n"
        f"c={c_plot}, alpha={alpha}, beta={beta}, L={L}, normalized X"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: q_t trajectory
    q_full = np.full(T, np.nan)
    q_full[K - 1 :] = aq["q_hist"]

    plt.figure(figsize=(10, 4))

    plt.plot(q_full)

    plt.axvline(
        Td,
        linestyle="--",
        label="abrupt drift",
    )

    plt.xlabel("Iteration")
    plt.ylabel(r"Adaptive $q_t$")

    plt.title(r"Adaptive $q_t$ trajectory")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drift_severity_impact(rows):
    """
    Generates graphs showing the impact of drift severity (c)
    on the improvement metrics of aq-JAPA relative to APA.
    """
    c_values = [row["c"] for row in rows]
    msd_imp = [row["%MSD_imp"] for row in rows]
    tau_gain = [row["tau_gain"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Improvement in MSD (%)
    color = 'tab:blue'
    ax1.set_xlabel('Shrinkage Parameter (c)')
    ax1.set_ylabel('MSD Improvement (%)', color=color)
    ax1.plot(c_values, msd_imp, marker='o', linestyle='-', color=color, linewidth=2, label='% MSD Imp')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    
    ax2 = ax1.twinx()  
    
    # Plot Gain across recovery iterations
    color = 'tab:red'
    ax2.set_ylabel(r'Recovery Time Gain ($\Delta \tau_{rec}$)', color=color)  
    ax2.plot(c_values, tau_gain, marker='s', linestyle='--', color=color, linewidth=2, label='Tau Gain')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("aq-JAPA Improvements vs. Drift Severity (Shrinkage factor c)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_rows = run_monte_carlo_experiment()
    plot_representative_run(c_plot=0.10, seed_plot=3)
    plot_drift_severity_impact(results_rows) 
    
    
    