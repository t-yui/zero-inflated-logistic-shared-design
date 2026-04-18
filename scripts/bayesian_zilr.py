import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import expit
from pypolyagamma import PyPolyaGamma
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import bernoulli, norm
from statsmodels.graphics.tsaplots import plot_acf
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


plt.rcParams["font.size"] = 16

eps = 1e-6


def save_figure(base_name):
    plt.savefig(f"{base_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{base_name}.png", bbox_inches="tight")


def generate_asymmetric_mislabeled_data(
    n, p, q, beta_0, gamma_0, r=0, common_indices=None
):
    X = np.zeros((n, p))
    Z = np.zeros((n, q))

    X[:, 0] = 1
    Z[:, 0] = 1

    if common_indices is not None:
        if isinstance(common_indices, tuple):
            common_indices = [common_indices]

        common_cols_X = set()
        common_cols_Z = set()

        for i, j in common_indices:
            if i > 0 and j > 0:
                i += 1
                j += 1
                if i < 1 or i > p:
                    raise ValueError(f"Column index i={i} in X is out of bounds.")
                if j < 1 or j > q:
                    raise ValueError(f"Column index j={j} in Z is out of bounds.")
                common_data = np.random.normal(size=n)
                X[:, i - 1] = common_data
                Z[:, j - 1] = common_data
                common_cols_X.add(i - 1)
                common_cols_Z.add(j - 1)

        for col in range(1, p):
            if col not in common_cols_X:
                X[:, col] = np.random.normal(size=n)

        for col in range(1, q):
            if col not in common_cols_Z:
                Z[:, col] = np.random.normal(size=n)
    else:
        if r > min(p, q) - 1:
            raise ValueError(
                "r must be less than or equal to min(p, q) - 1 (excluding intercept)."
            )

        common_columns = np.random.normal(size=(n, r))
        X_unique = np.random.normal(size=(n, p - r - 1))
        Z_unique = np.random.normal(size=(n, q - r - 1))

        X[:, 1 : 1 + r] = common_columns
        Z[:, 1 : 1 + r] = common_columns

        X[:, 1 + r :] = X_unique
        Z[:, 1 + r :] = Z_unique

    if len(beta_0) != p or len(gamma_0) != q:
        raise ValueError(
            "Length of beta_0 and gamma_0 must match p and q, respectively."
        )

    linear_combination = X @ beta_0
    pi_0 = 1 / (1 + np.exp(-linear_combination))
    y_0 = np.random.binomial(1, pi_0)

    kappa = np.exp(Z @ gamma_0)
    flip_prob = kappa / (kappa + 1)
    y = y_0.copy()
    for i in range(n):
        if y_0[i] == 1:
            y[i] = 1 - np.random.binomial(1, flip_prob[i])

    return X, Z, y_0, y, pi_0


def generate_asymmetric_mislabeled_data_binary(
    n, p, q, beta_0, gamma_0, r=0, common_indices=None, prob=0.5
):
    X = np.zeros((n, p))
    Z = np.zeros((n, q))
    X[:, 0] = 1
    Z[:, 0] = 1

    if common_indices is not None:
        if isinstance(common_indices, tuple):
            common_indices = [common_indices]
        common_cols_X = set()
        common_cols_Z = set()

        for i, j in common_indices:
            if i > 0 and j > 0:
                i += 1
                j += 1
                if i < 1 or i > p:
                    raise ValueError(f"Column index i={i} in X is out of bounds.")
                if j < 1 or j > q:
                    raise ValueError(f"Column index j={j} in Z is out of bounds.")
                common_data = np.random.binomial(1, prob, size=n)
                X[:, i - 1] = common_data
                Z[:, j - 1] = common_data
                common_cols_X.add(i - 1)
                common_cols_Z.add(j - 1)

        for col in range(1, p):
            if col not in common_cols_X:
                X[:, col] = np.random.binomial(1, prob, size=n)

        for col in range(1, q):
            if col not in common_cols_Z:
                Z[:, col] = np.random.binomial(1, prob, size=n)
    else:
        if r > min(p, q) - 1:
            raise ValueError(
                "r must be less than or equal to min(p, q) - 1 (excluding intercept)."
            )

        common_columns = np.random.binomial(1, prob, size=(n, r))
        X_unique = np.random.binomial(1, prob, size=(n, p - r - 1))
        Z_unique = np.random.binomial(1, prob, size=(n, q - r - 1))

        X[:, 1 : 1 + r] = common_columns
        Z[:, 1 : 1 + r] = common_columns

        X[:, 1 + r :] = X_unique
        Z[:, 1 + r :] = Z_unique

    if len(beta_0) != p or len(gamma_0) != q:
        raise ValueError(
            "Length of beta_0 and gamma_0 must match p and q, respectively."
        )

    linear_combination = X @ beta_0
    pi_0 = 1 / (1 + np.exp(-linear_combination))
    y_0 = np.random.binomial(1, pi_0, size=n)

    kappa = np.exp(Z @ gamma_0)
    flip_prob = kappa / (kappa + 1)
    y = y_0.copy()
    for i in range(n):
        if y_0[i] == 1:
            y[i] = 1 - np.random.binomial(1, flip_prob[i])

    return X, Z, y_0, y, pi_0


def compute_log_likelihood(y, X, Z, beta, gamma, h):
    ll = 0.0
    n = len(y)
    for i in range(n):
        z_i = Z[i]
        x_i = X[i]
        if y[i] == 1:
            ll += (
                -np.log(1 + np.exp(z_i.dot(gamma)))
                + x_i.dot(beta)
                - np.log(1 + np.exp(x_i.dot(beta)))
            )
        else:
            if h[i] == 1:
                ll += z_i.dot(gamma) - np.log(1 + np.exp(z_i.dot(gamma)))
            else:
                ll += -np.log(1 + np.exp(z_i.dot(gamma))) - np.log(
                    1 + np.exp(x_i.dot(beta))
                )
    return ll


class ChainState:
    def __init__(self, beta, gamma, h):
        self.beta = beta.copy()
        self.gamma = gamma.copy()
        self.h = h.copy()


def update_chain(state, y, X, Z, b0, B0, g0, G0, temperature, pg):
    n = len(y)
    p = X.shape[1]
    q = Z.shape[1]

    beta = state.beta.copy()
    gamma = state.gamma.copy()
    h = state.h.copy()

    h[y == 1] = 0

    idx0 = np.where(y == 0)[0]
    if idx0.size > 0:
        z_idx = Z[idx0]
        x_idx = X[idx0]
        A = np.exp(z_idx.dot(gamma))
        weight1 = (A / (1 + A)) ** (1.0 / temperature)
        weight0 = ((1.0 / (1 + A)) * (1.0 / (1 + np.exp(x_idx.dot(beta))))) ** (
            1.0 / temperature
        )
        prob_h = weight1 / (weight1 + weight0 + eps)
        prob_h = np.clip(prob_h, 0, 1)
        h[idx0] = np.random.binomial(1, prob_h)

    w_gamma = np.zeros(n)
    for i in range(n):
        w_gamma[i] = pg.pgdraw(1.0 / temperature, Z[i].dot(gamma))
    eta = (h - 0.5) / (temperature * w_gamma + eps)
    W_gamma = np.diag(w_gamma)
    invG0 = np.linalg.inv(G0)
    G1 = np.linalg.inv(Z.T @ W_gamma @ Z + invG0)
    g1 = G1 @ (Z.T @ (w_gamma * eta) + invG0 @ g0)
    gamma = multivariate_normal.rvs(mean=g1, cov=G1)

    idx_beta = np.where(h == 0)[0]
    if idx_beta.size > 0:
        X_star = X[idx_beta]
        y_star = y[idx_beta]
        m = len(idx_beta)
        omega = np.zeros(m)
        for j in range(m):
            omega[j] = pg.pgdraw(1.0 / temperature, X_star[j].dot(beta))
        xi = (y_star - 0.5) / (temperature * (omega + eps))
        Omega = np.diag(omega)
        invB0 = np.linalg.inv(B0)
        B1 = np.linalg.inv(X_star.T @ Omega @ X_star + invB0)
        b1 = B1 @ (X_star.T @ (omega * xi) + invB0 @ b0)
        beta = multivariate_normal.rvs(mean=b1, cov=B1)

    return ChainState(beta, gamma, h)


def parallel_tempering_gibbs_sampler(
    y,
    X,
    Z,
    b0,
    B0,
    g0,
    G0,
    temperatures,
    burn_in=2000,
    total_iterations=5000,
    swap_interval=50,
    seed=1,
):
    np.random.seed(seed)
    n = len(y)
    p = X.shape[1]
    q = Z.shape[1]
    M = len(temperatures)
    pg = PyPolyaGamma(seed=np.random.randint(1e5))

    chains = []
    for m in range(M):
        beta_init = np.zeros(p)
        gamma_init = np.zeros(q)
        h_init = np.zeros(n, dtype=int)
        chains.append(ChainState(beta_init, gamma_init, h_init))

    samples_beta = []
    samples_gamma = []

    total_iters = burn_in + total_iterations
    for it in range(total_iters):
        for m in range(M):
            T_m = temperatures[m]
            chains[m] = update_chain(chains[m], y, X, Z, b0, B0, g0, G0, T_m, pg)

        if it % swap_interval == 0:
            for m in range(M - 1):
                T_low = temperatures[m]
                T_high = temperatures[m + 1]
                ll_low = compute_log_likelihood(
                    y, X, Z, chains[m].beta, chains[m].gamma, chains[m].h
                )
                ll_high = compute_log_likelihood(
                    y, X, Z, chains[m + 1].beta, chains[m + 1].gamma, chains[m + 1].h
                )
                delta = (1.0 / T_low - 1.0 / T_high) * (ll_high - ll_low)

                if np.log(np.random.rand()) < delta:
                    chains[m].beta, chains[m + 1].beta = (
                        chains[m + 1].beta,
                        chains[m].beta,
                    )
                    chains[m].gamma, chains[m + 1].gamma = (
                        chains[m + 1].gamma,
                        chains[m].gamma,
                    )
                    chains[m].h, chains[m + 1].h = chains[m + 1].h, chains[m].h

        if it >= burn_in:
            samples_beta.append(chains[0].beta.copy())
            samples_gamma.append(-chains[0].gamma.copy())

    return np.array(samples_beta), np.array(samples_gamma)


def plot_trace(samples, parameter_name="parameter", file_prefix=None):
    num_param = samples.shape[1]
    for i in range(num_param):
        plt.figure(figsize=(10, 4))
        plt.plot(samples[:, i])
        plt.title(f"Trace plot of {parameter_name}_{i}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.tight_layout()
        if file_prefix is not None:
            save_figure(f"{file_prefix}_{parameter_name}_{i}")
        plt.show()


def plot_histogram(samples, parameter_name="parameter", bins=30, file_prefix=None):
    num_param = samples.shape[1]
    for i in range(num_param):
        plt.figure(figsize=(6, 4))
        plt.hist(samples[:, i], bins=bins, alpha=0.75)
        plt.title(f"Posterior histogram of {parameter_name}_{i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if file_prefix is not None:
            save_figure(f"{file_prefix}_{parameter_name}_{i}")
        plt.show()


def select_cluster_number_bic(samples, max_clusters=10):
    lowest_bic = np.infty
    best_n_clusters = 1
    bic_scores = []
    for n in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=1)
        gmm.fit(samples)
        bic = gmm.bic(samples)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_n_clusters = n
    return best_n_clusters, bic_scores


def summarize_multimodal_posterior(samples, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
    labels = kmeans.fit_predict(samples)

    summary = {}
    total_samples = samples.shape[0]
    for k in range(n_clusters):
        cluster_samples = samples[labels == k]
        n_cluster = cluster_samples.shape[0]
        mean = np.mean(cluster_samples, axis=0)
        cov = np.cov(cluster_samples, rowvar=False)
        cred_int = np.percentile(cluster_samples, [2.5, 97.5], axis=0)
        summary[k] = {
            "n_samples": n_cluster,
            "proportion": n_cluster / total_samples,
            "mean": mean,
            "covariance": cov,
            "cred_int": cred_int,
        }
    return labels, summary


def summarize_multimodal_posterior_gmm_bic(samples, min_components=1, max_components=2):
    best_bic = np.infty
    best_n_components = None
    best_gmm = None

    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=1)
        gmm.fit(samples)
        bic = gmm.bic(samples)
        if bic < best_bic:
            best_bic = bic
            best_n_components = n
            best_gmm = gmm

    labels = best_gmm.predict(samples)

    summary = {}
    total_samples = samples.shape[0]
    for k in range(best_n_components):
        cluster_samples = samples[labels == k]
        n_cluster = cluster_samples.shape[0]
        mean = np.mean(cluster_samples, axis=0)
        cov = np.cov(cluster_samples, rowvar=False)
        cred_int = np.percentile(cluster_samples, [2.5, 97.5], axis=0)
        summary[k] = {
            "n_samples": n_cluster,
            "proportion": n_cluster / total_samples,
            "mean": mean,
            "covariance": cov,
            "cred_int": cred_int,
        }

    return labels, summary


# simulation settings
np.random.seed(1)
n = 2000
p = 5
q = 5
beta_0 = np.array([0.5, 1, 0.5, 0.5, 0.25])
gamma_0 = np.array([-1.7, 1, 1, -0.5, -0.5])

# algorithm settings
random_state = 2025
b0 = np.zeros(p)
B0 = 100.0 * np.eye(p)
g0 = np.zeros(q)
G0 = 100.0 * np.eye(q)

num_replicas = 20
r = 1.05
temperatures = [r**i for i in range(num_replicas)]

total_iterations = 50000
swap_interval = 50
burn_in = 3000


# Continuous covariates

groups = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
np.random.seed(1)
x, z, y_0, y, pi_0i = generate_asymmetric_mislabeled_data(
    n, p, q, beta_0, gamma_0, common_indices=groups
)

samples_beta_1, samples_gamma_1 = parallel_tempering_gibbs_sampler(
    y,
    x,
    z,
    b0,
    B0,
    g0,
    G0,
    temperatures,
    burn_in=burn_in,
    total_iterations=total_iterations,
    swap_interval=swap_interval,
    seed=random_state,
)

plot_trace(samples_beta_1, parameter_name="beta", file_prefix="continuous_trace")
plot_trace(samples_gamma_1, parameter_name="gamma", file_prefix="continuous_trace")

plot_histogram(samples_beta_1, parameter_name="beta", file_prefix="continuous_hist")
plot_histogram(samples_gamma_1, parameter_name="gamma", file_prefix="continuous_hist")

print(np.mean(samples_beta_1, axis=0))
print(np.mean(samples_gamma_1, axis=0))

samples = np.concatenate([samples_beta_1, samples_gamma_1], axis=1)
labels, summary = summarize_multimodal_posterior(samples, n_clusters=2)
for k in summary:
    print(f"Cluster {k}:")
    print("  Number of samples =", summary[k]["n_samples"])
    print("  Proportion =", summary[k]["proportion"])
    print("  Mean =", summary[k]["mean"])
    print("  Covariance matrix =\n", summary[k]["covariance"])
    print("  95% credible interval (2.5% and 97.5%) =", summary[k]["cred_int"])
    print()

pca = PCA(n_components=2)
samples_2d = pca.fit_transform(samples)
plt.figure(figsize=(8, 6))
plt.scatter(samples_2d[:, 0], samples_2d[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("PC1", fontsize=24)
plt.ylabel("PC2", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
save_figure("post_1_1")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("beta_0")
plt.ylabel("beta_1")
# plt.title("Clustered MCMC samples")
plt.tight_layout()
save_figure("post_1_1_beta0_beta1")
plt.show()


# Binary covariates

groups = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
np.random.seed(1)
x, z, y_0, y, pi_0i = generate_asymmetric_mislabeled_data_binary(
    n, p, q, beta_0, gamma_0, common_indices=groups
)

samples_beta_3, samples_gamma_3 = parallel_tempering_gibbs_sampler(
    y,
    x,
    z,
    b0,
    B0,
    g0,
    G0,
    temperatures,
    burn_in=burn_in,
    total_iterations=total_iterations,
    swap_interval=swap_interval,
    seed=random_state,
)

plot_trace(samples_beta_3, parameter_name="beta", file_prefix="binary_trace")
plot_trace(samples_gamma_3, parameter_name="gamma", file_prefix="binary_trace")

plot_histogram(samples_beta_3, parameter_name="beta", file_prefix="binary_hist")
plot_histogram(samples_gamma_3, parameter_name="gamma", file_prefix="binary_hist")

print(np.mean(samples_beta_3, axis=0))
print(np.mean(samples_gamma_3, axis=0))

samples = np.concatenate([samples_beta_3, samples_gamma_3], axis=1)
labels, summary = summarize_multimodal_posterior(samples, n_clusters=2)
for k in summary:
    print(f"Cluster {k}:")
    print("  Number of samples =", summary[k]["n_samples"])
    print("  Proportion =", summary[k]["proportion"])
    print("  Mean =", summary[k]["mean"])
    print("  Covariance matrix =\n", summary[k]["covariance"])
    print("  95% credible interval (2.5% and 97.5%) =", summary[k]["cred_int"])
    print()

pca = PCA(n_components=2)
samples_2d = pca.fit_transform(samples)
plt.figure(figsize=(8, 6))
plt.scatter(samples_2d[:, 0], samples_2d[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("PC1", fontsize=24)
plt.ylabel("PC2", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
save_figure("post_3_1")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("beta_0")
plt.ylabel("beta_1")
# plt.title("Clustered MCMC samples")
plt.tight_layout()
save_figure("post_3_1_beta0_beta1")
plt.show()


# Mixed covariates with X=Z


def generate_asymmetric_mislabeled_data_mixed(
    n,
    p,
    q,
    beta_0,
    gamma_0,
    x_dist=None,
    z_dist=None,
    r=0,
    common_indices=None,
    prob=0.5,
):
    if x_dist is None:
        x_dist = ["normal"] * p
    if z_dist is None:
        z_dist = ["normal"] * q
    if len(x_dist) != p or len(z_dist) != q:
        raise ValueError("The lengths of x_dist and z_dist must match p and q.")

    X = np.zeros((n, p))
    Z = np.zeros((n, q))
    X[:, 0] = 1
    Z[:, 0] = 1

    def generate_column(dist_type):
        if dist_type == "normal":
            return np.random.normal(size=n)
        if dist_type == "binary":
            return np.random.binomial(1, prob, size=n)
        raise ValueError(f"Unknown distribution type: {dist_type}")

    if common_indices is not None:
        if isinstance(common_indices, tuple):
            common_indices = [common_indices]
        common_cols_X = set()
        common_cols_Z = set()
        for i, j in common_indices:
            if i < 1 or j < 1:
                raise ValueError(
                    "Each element of common_indices must specify non-intercept columns (>= 1)."
                )
            col_X = i
            col_Z = j
            if col_X >= p:
                raise ValueError(f"Column index i={i} in X exceeds p.")
            if col_Z >= q:
                raise ValueError(f"Column index j={j} in Z exceeds q.")
            if x_dist[col_X] != z_dist[col_Z]:
                raise ValueError(
                    f"Common column (X: {i}, Z: {j}) has inconsistent distributions: "
                    f"{x_dist[col_X]} vs {z_dist[col_Z]}."
                )
            common_data = generate_column(x_dist[col_X])
            X[:, col_X] = common_data
            Z[:, col_Z] = common_data
            common_cols_X.add(col_X)
            common_cols_Z.add(col_Z)

        for col in range(1, p):
            if col not in common_cols_X:
                X[:, col] = generate_column(x_dist[col])
        for col in range(1, q):
            if col not in common_cols_Z:
                Z[:, col] = generate_column(z_dist[col])
    else:
        if r > min(p - 1, q - 1):
            raise ValueError(
                "r must be less than or equal to min(p - 1, q - 1) (excluding the intercept)."
            )
        for col in range(1, 1 + r):
            if x_dist[col] != z_dist[col]:
                raise ValueError(
                    f"Column {col} specified as common has inconsistent distributions: "
                    f"{x_dist[col]} vs {z_dist[col]}."
                )
            common_data = generate_column(x_dist[col])
            X[:, col] = common_data
            Z[:, col] = common_data
        for col in range(1 + r, p):
            X[:, col] = generate_column(x_dist[col])
        for col in range(1 + r, q):
            Z[:, col] = generate_column(z_dist[col])

    if len(beta_0) != p or len(gamma_0) != q:
        raise ValueError("The lengths of beta_0 and gamma_0 must match p and q.")

    linear_combination = X @ beta_0
    pi_0 = 1 / (1 + np.exp(-linear_combination))
    y_0 = np.random.binomial(1, pi_0)

    kappa = np.exp(Z @ gamma_0)
    flip_prob = kappa / (kappa + 1)
    y = y_0.copy()
    for i in range(n):
        if y_0[i] == 1:
            y[i] = 1 - np.random.binomial(1, flip_prob[i])

    return X, Z, y_0, y, pi_0


groups = [(1, 1), (2, 2), (3, 3), (4, 4)]
x_dist = ["normal", "normal", "binary", "binary", "binary"]
z_dist = ["normal", "normal", "binary", "binary", "binary"]
np.random.seed(1)
x, z, y_0, y, pi_0i = generate_asymmetric_mislabeled_data_mixed(
    n, p, q, beta_0, gamma_0, x_dist=x_dist, z_dist=z_dist, common_indices=groups
)

samples_beta_5, samples_gamma_5 = parallel_tempering_gibbs_sampler(
    y,
    x,
    z,
    b0,
    B0,
    g0,
    G0,
    temperatures,
    burn_in=burn_in,
    total_iterations=total_iterations,
    swap_interval=swap_interval,
    seed=random_state,
)

plot_trace(samples_beta_5, parameter_name="beta", file_prefix="mixed_trace")
plot_trace(samples_gamma_5, parameter_name="gamma", file_prefix="mixed_trace")

plot_histogram(samples_beta_5, parameter_name="beta", file_prefix="mixed_hist")
plot_histogram(samples_gamma_5, parameter_name="gamma", file_prefix="mixed_hist")

print(np.mean(samples_beta_5, axis=0))
print(np.mean(samples_gamma_5, axis=0))

samples = np.concatenate([samples_beta_5, samples_gamma_5], axis=1)
labels, summary = summarize_multimodal_posterior(samples, n_clusters=2)
for k in summary:
    print(f"Cluster {k}:")
    print("  Number of samples =", summary[k]["n_samples"])
    print("  Proportion =", summary[k]["proportion"])
    print("  Mean =", summary[k]["mean"])
    print("  Covariance matrix =\n", summary[k]["covariance"])
    print("  95% credible interval (2.5% and 97.5%) =", summary[k]["cred_int"])
    print()

pca = PCA(n_components=2)
samples_2d = pca.fit_transform(samples)
plt.figure(figsize=(8, 6))
plt.scatter(samples_2d[:, 0], samples_2d[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("PC1", fontsize=24)
plt.ylabel("PC2", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
save_figure("post_5_1")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.xlabel("beta_0")
plt.ylabel("beta_1")
# plt.title("Clustered MCMC samples")
plt.tight_layout()
save_figure("post_5_1_beta0_beta1")
plt.show()
