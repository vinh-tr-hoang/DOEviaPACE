"""
This code implements methods to compute the total expected conditional variance (tECV)
for three different techniques:
 - Projection-based Approximation of the Conditional Expectation (PACE),
 - PACE with data augmentation, and
 - Importance Sampling (IS),
for the one-dimensional linear observational map. For 1D, tECV is just ECV.

The script creates synthetic datasets and evaluates the performance of these methods
by computing the mean absolute error (MAE) between estimated and reference tECV values.

The performance of each method is plotted in terms of the relative mean absolute
error (relMAE) against the number of samples used.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


#--- Functions related to PACE-based approach ---#
def _pace_ecv(training_data: tuple[np.ndarray, np.ndarray], evaluating_data: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute the tECV based on PACE
    (Projection-based Approximation of the Conditional Expectation).

    Parameters:
        training_data (tuple[np.ndarray, np.ndarray]): tuple containing the dataset for training,
        with two elements:
            - training_data[0]: Array-like object of samples of the identifying random vector.
            - training_data[1]: Array-like object of corresponding observations.
        evaluating_data (tuple[np.ndarray, np.ndarray]): tuple containing the dataset for evaluation
        with the same structure as training_data.

    Returns:
        float: tECV computed based on PACE.
    """

    y_mean = np.mean(training_data[1])
    q_mean = np.mean(training_data[0])
    cov_y = np.mean(training_data[1] * training_data[1]) - y_mean ** 2
    cov_q_y = np.mean((training_data[1] - y_mean) * (training_data[0] - q_mean))

    # Fit the PACE (linear) coefficients, then evaluate on the evaluation set
    a_pace = cov_q_y / cov_y
    b_pace = q_mean - np.mean(a_pace * training_data[1])
    pace_ce = a_pace * evaluating_data[1] + b_pace
    return np.mean((evaluating_data[0] - pace_ce) ** 2)


def pace_ecv(d: float, N: int, M: int) -> float:
    """
    Compute the tECV based on PACE
    without data augmentation.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the training dataset.
        M (int): Number of samples in the evaluation dataset.

    Returns:
        float: tECV computed based on PACE.
    """
    training_data = create_dataset(d=d, N=N)
    evaluating_data = create_dataset(d=d, N=M)
    return _pace_ecv(training_data, evaluating_data)


def data_augmented_pace_ecv(d: float, N: int, M: int) -> float:
    """
    Compute the tECV based on PACE with data augmentation.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the training dataset.
        M (int): Number of samples in the evaluation dataset.

    Returns:
        float: tECV computed based on PACE.
    """
    training_data = create_augmented_dataset(d=d, N=N)
    evaluating_data = create_augmented_dataset(d=d, N=M)
    return _pace_ecv(training_data, evaluating_data)


#--- Important sampling-based approach ---#
def importance_sampling_ecv(d: float, N_inner: int) -> float:
    """
    Compute the tECV using
    the Importance Sampling (IS) method.

    Parameters:
        d (float): Design parameter.
        N_inner (int): Number of samples of the inner loop.

    Returns:
        float: tECV computed using the IS method.
    """
    # Outer dataset with size of 1
    Do = create_dataset(d=d, N=1)
    y = Do[1][0]

    # Inner dataset with size of N_inner
    q_samples = q_rv.rvs(size=N_inner)
    y_samples = noise_free_observational_map(q=q_samples, d=d)

    likelihood = epsilon_rv.pdf(y_samples - y)

    # Normalize the likelihood to improve the numerical stability
    likelihood_mean = np.mean(likelihood)
    if likelihood_mean == 0.:
        return np.nan
    likelihood = likelihood / likelihood_mean

    # Importance-sampling posterior mean and variance
    posterior_mean = np.mean(q_samples * likelihood)
    posterior_var = np.mean((q_samples - posterior_mean) ** 2 * likelihood)
    return posterior_var


#--- Functions related to observational model ---#
def linear_observation_coefficient(d: float) -> float:
    """
    Return the linear observational map's coefficient c
        h(q) = c*q
    Parameters:
        d (float): Design parameter used to calculate the coefficient.

    Returns:
        float: Coefficient c of the linear observational map.
    """
    return 1. / ((d - 0.5) ** 2 + 1)


def noise_free_observational_map(q, d: float) -> float:
    """
    Compute the noise-free observational map.

    Parameters:
        q (float or np.ndarray): Samples of the identifying random vector.
        d (float): Design parameter used in the noise-free map.

    Returns:
        float or np.ndarray: Noise-free observational map.
    """
    return q * linear_observation_coefficient(d)


def observational_map(q, d: float) -> float:
    """
    Compute the observational map by adding noise to the noise-free map.

    Parameters:
        q (float or np.ndarray): Samples of the identifying random vector.
        d (float): Design parameter used in the noise-free map.

    Returns:
        float or np.ndarray: Observational map obtained by adding noise to the noise-free map.
    """
    return noise_free_observational_map(q=q, d=d) + epsilon_rv.rvs(size=q.size)


def ref_ecv(d: float) -> float:
    """
    Compute the reference tECV.

    Parameters:
        d (float): Design parameter.

    Returns:
        float: Reference tECV.
    """
    h = 1. / ((d - 0.5) ** 2 + 1)
    cov_y = h ** 2 * sigma_q ** 2 + sigma_epsilon ** 2
    cov_q_y = h * sigma_q ** 2
    k = cov_q_y / cov_y

    return sigma_q ** 2 - 2 * k * cov_q_y + k ** 2 * cov_y


def create_augmented_dataset(d: float, N: int, Na: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an augmented dataset by generating additional samples with added noise.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the base dataset.
        Na (int): Number of times to replicate the base dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the augmented dataset
        with two elements:
            - q_samples: Array-like object of samples of the identifying random vector.
            - y_samples: Array-like object of augmented noisy observations.
    """
    q_samples_base = q_rv.rvs(size=N)
    y_samples_base = noise_free_observational_map(q=q_samples_base, d=d)
    q_samples = np.repeat(q_samples_base, Na)
    y_samples = np.repeat(y_samples_base, Na) + epsilon_rv.rvs(size=N * Na)
    return q_samples, y_samples


def create_dataset(d: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of samples by generating observations based on a noise-free observational map.

    Parameters:
        d (float): Design parameter used in the noise-free observational map.
        N (int): Number of samples to generate in the dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the dataset with two elements:
            - q_samples: Array-like object of samples of the identifying random vector.
            - y_samples: Array-like object of corresponding observations.
    """
    q_samples = q_rv.rvs(size=N)
    y_samples = observational_map(q=q_samples, d=d)
    return q_samples, y_samples


def plot_expected_conditional_variance_global_approach(n=5000, number_runs=2):
    """
    Plot the statistical errors in estimating the tECV
     for different methods.

    Parameters:
        n (int): Total number of samples.
        number_runs (int): Number of runs.

    Returns:
        None
    """
    d_array = np.linspace(0, 1, 10)
    ecv_pace, ecv_is = np.zeros((number_runs, d_array.size)), np.zeros((number_runs, d_array.size))
    reference_ecv = np.zeros_like(d_array)
    for i in range(d_array.size):
        reference_ecv[i] = ref_ecv(d_array[i])

    # IS estimate over number_runs; resample on NaN or > 4x reference (divide-by-zero guard)
    for r in range(number_runs):
        for i in range(d_array.size):
            _ecv = importance_sampling_ecv(d=d_array[i], N_inner=n)
            _retries = 0
            while (np.isnan(_ecv) or _ecv > reference_ecv[i] * 4) and _retries < 100:
                _ecv = importance_sampling_ecv(d=d_array[i], N_inner=n)
                _retries += 1
            ecv_is[r, i] = _ecv

    # PACE (with augmentation) estimate over number_runs
    for r in range(number_runs):
        ecv_pace[r, :] = [data_augmented_pace_ecv(d=d_, N=n // 2, M=n // 2) for d_ in d_array]

    # Plot statistical_error_distribution of PACE-based estimator
    _yscale = 10 ** 3
    plt.figure()
    plt.plot(d_array, reference_ecv * _yscale, '-', label='reference')
    plt.errorbar(d_array, ecv_pace.mean(axis=0) * _yscale, ecv_pace.std(axis=0) * _yscale, alpha=.75,
                 label=r'PACE-based ($N+M=$ ' + str(n) + ')',
                 ls='-', marker='s', markersize=5, capsize=3, capthick=1)
    x, y, e = d_array, ecv_pace.mean(axis=0) * _yscale, ecv_pace.std(axis=0) * _yscale
    data = {
        'x': x,
        'y1': [yi - ei for yi, ei in zip(y, e)],
        'y2': [yi + ei for yi, ei in zip(y, e)]}
    plt.fill_between(**data, alpha=.25)
    plt.xlabel(r'$d$')
    plt.ylabel(r'ECV $\times 10^{3}$')
    plt.legend()
    plt.ylim([0.00008 * _yscale, 0.000189 * _yscale])
    plt.savefig('PACE_statistical_error_dist.pdf')
    plt.show()

    # Plot statistical_error_distribution of important sampling-based estimator
    plt.figure()
    plt.plot(d_array, reference_ecv * _yscale, '-', label='reference')
    plt.errorbar(d_array, ecv_is.mean(axis=0) * _yscale, ecv_is.std(axis=0) * _yscale, alpha=.75,
                 label=r'IS-based ($N_{{i}}$=' + str(n) + r', $N_{{o}}$=1' + ')',
                 ls=':', marker='s', markersize=5, capsize=3, capthick=1)
    x, y, e = d_array, ecv_is.mean(axis=0) * _yscale, ecv_is.std(axis=0) * _yscale
    data = {
        'x': x,
        'y1': [yi - ei for yi, ei in zip(y, e)],
        'y2': [yi + ei for yi, ei in zip(y, e)]}

    plt.fill_between(**data, alpha=.25)
    plt.xlabel(r'$d$')
    plt.ylabel(r'ECV $\times 10^{3}$')
    plt.legend()
    plt.ylim([0.00008 * _yscale, 0.000189 * _yscale])
    plt.savefig('IS_statistical_error_dist.pdf')


if __name__ == '__main__':

    sigma_q = 2           # std of the identifying random variable
    sigma_epsilon = 1e-2  # std of the observational errors

    q_rv = norm(loc=0, scale=sigma_q)
    epsilon_rv = norm(loc=0, scale=sigma_epsilon)

    # Figure 1: statistical-error distribution of the ECV estimators vs d
    plot_expected_conditional_variance_global_approach(n=10000, number_runs=1000)

    d = 0.5                                             # design parameter under test
    N = np.array([100, 500, 1000, 5000, 10000, 20000]) # PACE training/eval sample sizes
    M = N
    N_inner = [200, 500, 1000, 5000, 10000, 40000]     # IS inner-loop sample sizes
    number_runs = 100

    reference_ecv = ref_ecv(d=d)

    ecv_pace = np.zeros(shape=(len(N), number_runs))
    ecv_pace_aug = np.zeros_like(ecv_pace)
    ecv_is = np.zeros(shape=(len(N_inner), number_runs))

    # Estimate ECV over number_runs repetitions: PACE, PACE + augmentation, and IS
    for i in range(len(N)):
        ecv_pace[i, :] = np.array([pace_ecv(d=d, N=N[i], M=M[i]) for _ in range(number_runs)])
        ecv_pace_aug[i, :] = np.array([data_augmented_pace_ecv(d=d, N=N[i], M=M[i]) for _ in range(number_runs)])
    for i in range(len(N_inner)):
        ecv_is[i, :] = np.array([importance_sampling_ecv(d=d, N_inner=N_inner[i]) for _ in range(number_runs)])

    # Mean absolute error vs the reference ECV
    l1error_pace = np.mean(np.abs(ecv_pace - reference_ecv), axis=1)
    l1error_pace_aug = np.mean(np.abs(ecv_pace_aug - reference_ecv), axis=1)
    l1error_is = np.mean(np.abs(ecv_is - reference_ecv), axis=1)

    plt.figure('relative MSE')
    plt.plot(N_inner, l1error_is / reference_ecv, '-x', label='IS-based')
    plt.plot(N + M, l1error_pace / reference_ecv, '-s', label='PACE-based')
    plt.plot(N + M, l1error_pace_aug / reference_ecv, '--s', label='PACE-based + data augmentation')
    plt.plot(N + M, 2 * np.sqrt(2. / N), '--k', linewidth=2,
             label=r'$\sqrt{{2}/{N}}+\sqrt{{2}/{M}}$')  # theoretical bound

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of samples')
    plt.ylabel(r'relMAE')
    plt.legend()
    plt.ylim([1e-3, 10])

    plt.savefig('linear_gaussian_1d_sigma_eps_' + str(sigma_epsilon) + '.pdf')
    plt.show()
