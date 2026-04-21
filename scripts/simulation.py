import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

np.random.seed(1)


def logistic(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def lr_log_likelihood(beta, X, y):
    eta = X @ beta
    p = logistic(eta)
    ll = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    return -ll


def lr_gradient(beta, X, y):
    eta = X @ beta
    p = logistic(eta)
    grad = X.T @ (y - p)
    return -grad


def fit_lr(X, y, init_params=None, max_iter=1000):
    d = X.shape[1]
    if init_params is None:
        init_params = np.random.randn(d) * 0.1
    result = minimize(
        lr_log_likelihood,
        init_params,
        args=(X, y),
        method="L-BFGS-B",
        jac=lr_gradient,
        options={"maxiter": max_iter},
    )
    if result.success and result.nit < max_iter:
        return result.x, result.fun, True
    return None, np.inf, False


def zilr_log_likelihood(params, X, y):
    d = X.shape[1]
    beta, gamma = params[:d], params[d:]
    p_beta, p_gamma = logistic(X @ beta), logistic(X @ gamma)
    p_combined = p_gamma * p_beta
    ll = np.sum(
        y * np.log(p_combined + 1e-10) + (1 - y) * np.log(1 - p_combined + 1e-10)
    )
    return -ll


def zilr_gradient(params, X, y):
    d = X.shape[1]
    beta, gamma = params[:d], params[d:]
    p_beta, p_gamma = logistic(X @ beta), logistic(X @ gamma)
    p_combined = p_gamma * p_beta
    grad_beta = X.T @ (
        y * p_gamma * p_beta * (1 - p_beta) / (p_combined + 1e-10)
        - (1 - y) * p_gamma * p_beta * (1 - p_beta) / (1 - p_combined + 1e-10)
    )
    grad_gamma = X.T @ (
        y * p_gamma * (1 - p_gamma) * p_beta / (p_combined + 1e-10)
        - (1 - y) * p_gamma * (1 - p_gamma) * p_beta / (1 - p_combined + 1e-10)
    )
    return -np.concatenate([grad_beta, grad_gamma])


def fit_zilr(X, y, init_params=None, max_iter=1000):
    d = X.shape[1]
    if init_params is None:
        init_params = np.random.randn(2 * d) * 0.1
    result = minimize(
        zilr_log_likelihood,
        init_params,
        args=(X, y),
        method="L-BFGS-B",
        jac=zilr_gradient,
        options={"maxiter": max_iter},
    )
    if result.success and result.nit < max_iter:
        return result.x[:d], result.x[d:], result.fun, True
    return None, None, np.inf, False


def check_reasonable_estimates(
    beta_est, gamma_est, beta_true, gamma_true, threshold=10
):
    for i in range(len(beta_true)):
        val = abs(beta_est[i])
        ref = abs(beta_true[i])
        if (ref > 0 and val > threshold * ref) or (ref == 0 and val > threshold):
            return False
    for i in range(len(gamma_true)):
        val = abs(gamma_est[i])
        ref = abs(gamma_true[i])
        if (ref > 0 and val > threshold * ref) or (ref == 0 and val > threshold):
            return False
    return True


def proposed_relabeling_method(
    X, y, beta_true, gamma_true, max_iter=1000, threshold=10
):
    beta_lr, _, lr_conv = fit_lr(X, y, max_iter=max_iter)
    if not lr_conv or beta_lr is None:
        return None, None, False
    beta1, gamma1, _, conv = fit_zilr(X, y, max_iter=max_iter)
    if not conv or beta1 is None:
        return None, None, False
    beta2, gamma2 = gamma1, beta1
    if np.linalg.norm(beta1 - beta_lr) <= np.linalg.norm(beta2 - beta_lr):
        if check_reasonable_estimates(beta1, gamma1, beta_true, gamma_true, threshold):
            return beta1, gamma1, True
    else:
        if check_reasonable_estimates(beta2, gamma2, beta_true, gamma_true, threshold):
            return beta2, gamma2, True
    return None, None, False


def generate_data(n, beta_true, gamma_true):
    d = len(beta_true)
    X = np.ones((n, d))
    X[:, 1:] = np.random.randn(n, d - 1)
    p_combined = logistic(X @ gamma_true) * logistic(X @ beta_true)
    y = np.random.binomial(1, p_combined)
    return X, y


def calculate_mislabel_rate(gamma_params, n_samples=10000):
    np.random.seed(123)
    X_temp = np.ones((n_samples, len(gamma_params)))
    X_temp[:, 1:] = np.random.randn(n_samples, len(gamma_params) - 1)
    return 1 - np.mean(logistic(X_temp @ gamma_params))


def calculate_metrics(estimates, true_params):
    estimates = np.array(estimates)
    return np.mean(estimates - true_params, axis=0), np.std(estimates, axis=0)


def save_dual_format(filename_base):
    for fmt in ["png", "pdf"]:
        plt.savefig(f"{filename_base}.{fmt}", dpi=300, bbox_inches="tight")


base_beta = np.array([0.5, 1.0, 0.5, 0.5, 0.25])
scenarios = {
    "Very Low Mislabel": {
        "beta": base_beta.copy(),
        "gamma": np.array([4.3, -1.0, -1.0, 0.5, 0.5]),
    },
    "Low Mislabel": {
        "beta": base_beta.copy(),
        "gamma": np.array([3.0, -1.0, -1.0, 0.5, 0.5]),
    },
    "Moderate Mislabel": {
        "beta": base_beta.copy(),
        "gamma": np.array([1.7, -1.0, -1.0, 0.5, 0.5]),
    },
    "High Mislabel": {
        "beta": base_beta.copy(),
        "gamma": np.array([1.0, -1.0, -1.0, 0.5, 0.5]),
    },
}

n_sim, n_sample = 10000, 1000
max_iter, threshold = 1000, 10
results = {
    s: {m: {"beta": [], "gamma": []} for m in ["proposed", "lr", "naive"]}
    for s in scenarios
}
convergence_stats = {
    s: {
        m: 0
        for m in [
            "proposed",
            "lr",
            "naive",
            "total",
            "proposed_unreasonable",
            "lr_unreasonable",
            "naive_unreasonable",
        ]
    }
    for s in scenarios
}

for s_name, params in scenarios.items():
    beta_true, gamma_true = params["beta"], params["gamma"]
    convergence_stats[s_name]["total"] = n_sim
    for _ in tqdm(range(n_sim), desc="Scenario: {0}".format(s_name)):
        X, y = generate_data(n_sample, beta_true, gamma_true)

        b_p, g_p, c_p = proposed_relabeling_method(
            X, y, beta_true, gamma_true, max_iter, threshold
        )
        if c_p and b_p is not None:
            results[s_name]["proposed"]["beta"].append(b_p)
            results[s_name]["proposed"]["gamma"].append(g_p)
            convergence_stats[s_name]["proposed"] += 1
        elif b_p is None and not c_p:
            b_t, g_t, c_t = proposed_relabeling_method(
                X, y, beta_true, gamma_true, max_iter, float("inf")
            )
            if c_t:
                convergence_stats[s_name]["proposed_unreasonable"] += 1

        b_lr, _, c_lr = fit_lr(X, y, max_iter=max_iter)
        if c_lr and b_lr is not None:
            if all(
                (
                    abs(b_lr[i]) <= threshold * abs(beta_true[i])
                    if abs(beta_true[i]) > 0
                    else abs(b_lr[i]) <= threshold
                )
                for i in range(len(beta_true))
            ):
                results[s_name]["lr"]["beta"].append(b_lr)
                convergence_stats[s_name]["lr"] += 1
            else:
                convergence_stats[s_name]["lr_unreasonable"] += 1

        b_n, g_n, _, c_n = fit_zilr(X, y, max_iter=max_iter)
        if c_n and b_n is not None:
            if check_reasonable_estimates(b_n, g_n, beta_true, gamma_true, threshold):
                results[s_name]["naive"]["beta"].append(b_n)
                results[s_name]["naive"]["gamma"].append(g_n)
                convergence_stats[s_name]["naive"] += 1
            else:
                convergence_stats[s_name]["naive_unreasonable"] += 1

latex_beta = "\\begin{table}[htbp]\n\\centering\n\\caption{Simulation Results: Parameter Estimation Performance}\n\\label{tab:simulation_results}\n\\begin{tabular}{llrrrrr}\n\\toprule\n\\multirow{2}{*}{Scenario} & \\multirow{2}{*}{Method} & \\multicolumn{5}{c}{Bias (SD)} \\\\\n\\cmidrule{3-7}\n& & $\\beta_0$ & $\\beta_1$ & $\\beta_2$ & $\\beta_3$ & $\\beta_4$ \\\\\n\\midrule\n"
for s_name, params in scenarios.items():
    b_true = params["beta"]
    for m in ["proposed", "lr", "naive"]:
        if results[s_name][m]["beta"]:
            bias, std = calculate_metrics(results[s_name][m]["beta"], b_true)
            row_start = f"\\multirow{{3}}{{*}}{{{s_name}}}" if m == "proposed" else ""
            m_name = {
                "proposed": "Proposed",
                "lr": "Standard LR",
                "naive": "Naive ZILR",
            }[m]
            latex_beta += (
                f"{row_start} & {m_name} & "
                + " & ".join(
                    [f"{bias[i]:.3f} ({std[i]:.3f})" for i in range(len(b_true))]
                )
                + " \\\\\n"
            )
    latex_beta += "\\midrule\n"
print(latex_beta.rstrip("\\midrule\n") + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")

latex_gamma = "\\begin{table}[htbp]\n\\centering\n\\caption{Simulation Results: Zero-Inflation Parameters ($\\gamma$)}\n\\label{tab:simulation_results_gamma}\n\\begin{tabular}{llrrrrr}\n\\toprule\n\\multirow{2}{*}{Scenario} & \\multirow{2}{*}{Method} & \\multicolumn{5}{c}{Bias (SD)} \\\\\n\\cmidrule{3-7}\n& & $\\gamma_0$ & $\\gamma_1$ & $\\gamma_2$ & $\\gamma_3$ & $\\gamma_4$ \\\\\n\\midrule\n"
for s_name, params in scenarios.items():
    g_true = params["gamma"]
    for m in ["proposed", "naive"]:
        if results[s_name][m]["gamma"]:
            bias, std = calculate_metrics(results[s_name][m]["gamma"], g_true)
            row_start = f"\\multirow{{2}}{{*}}{{{s_name}}}" if m == "proposed" else ""
            m_name = "Proposed" if m == "proposed" else "Naive ZILR"
            latex_gamma += (
                f"{row_start} & {m_name} & "
                + " & ".join(
                    [f"{bias[i]:.3f} ({std[i]:.3f})" for i in range(len(g_true))]
                )
                + " \\\\\n"
            )
    latex_gamma += "\\midrule\n"
print(
    latex_gamma.rstrip("\\midrule\n") + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)


def generate_boxplot(data_type, bias=True):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for idx, (s_name, params) in enumerate(scenarios.items()):
        ax = axes[idx]
        true_vals = params[data_type]
        sym = "\\beta" if data_type == "beta" else "\\gamma"
        plot_data = []
        for m in (
            ["proposed", "lr", "naive"]
            if data_type == "beta"
            else ["proposed", "naive"]
        ):
            if results[s_name][m][data_type]:
                estimates = np.array(results[s_name][m][data_type])
                m_lbl = {"proposed": "Prop", "lr": "LR", "naive": "Naive"}[m]
                for i in range(len(true_vals)):
                    vals = estimates[:, i] - true_vals[i] if bias else estimates[:, i]
                    plot_data.extend(zip(vals, [f"${sym}_{i}$ ({m_lbl})"] * len(vals), [f"{m_lbl}"] * len(vals)))
        df = pd.DataFrame(plot_data, columns=["Val", "Param", "method"])
        if data_type == "beta":
            sns.boxplot(data=df, x="Param", y="Val", ax=ax, hue="method", dodge=False)
        else:
            sns.boxplot(data=df, x="Param", y="Val", ax=ax, hue="method", dodge=False)
        if bias:
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        else:
            for i, label in enumerate(ax.get_xticklabels()):
                p_idx = int(label.get_text().split(" ")[0][1:])
                ax.axhline(
                    y=true_vals[p_idx],
                    xmin=(i - 0.4) / len(ax.get_xticklabels()),
                    xmax=(i + 0.4) / len(ax.get_xticklabels()),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                )
        ax.set_title(
            f'{s_name} - {data_type.capitalize()} {"Bias" if bias else "Estimates"}'
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    save_dual_format(f'parameter_{data_type}_{"bias" if bias else "estimate"}_boxplots')
    plt.show()


generate_boxplot("beta", True)
generate_boxplot("gamma", True)

print("\nSummary Statistics:\n" + "=" * 90)
print(
    f"{'Scenario':<20} {'Method':<15} {'Converged':<10} {'Unreason.':<10} {'Fail':<10} {'Total':<10} {'Rate':<10}"
)
for s in scenarios:
    for m in ["proposed", "lr", "naive"]:
        conv, unr, tot = (
            convergence_stats[s][m],
            convergence_stats[s][f"{m}_unreasonable"],
            convergence_stats[s]["total"],
        )
        print(
            f"{s if m=='proposed' else '':<20} {m:<15} {conv:<10} {unr:<10} {tot-conv-unr:<10} {tot:<10} {conv/tot*100:<10.1f}%"
        )
    print("-" * 90)
