"""
This code implements methods to compute the total expected conditional variance (tECV)
for three different techniques:
 - Projection-based Approximation of the Conditional Expectation (PACE),
 - PACE with data augmentation, and 
 - Importance Sampling (IS),
for the multidimensional linear observational map.

The script creates synthetic datasets and evaluates the performance of these methods 
by computing the mean absolute error (MAE) between estimated and reference tECV values.

The performance of each method is plotted in terms of the relative mean absolute
error (relMAE) against the number of samples used.
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


#--- Functions related to PACE-based approach ---#
def _pace_ecv(training_data: tuple[np.ndarray, np.ndarray], evaluating_data: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute the tECV based on PACE
    (Projection-based Approximation of the Conditional Expectation).

    Parameters:
        training_data (Tuple[np.ndarray, np.ndarray]): tuple containing the dataset for training,
        with two elements:
            - training_data[0]: Matrix-like object of samples of the identifying random vector.
            - training_data[1]: Matrix-like object of corresponding observations.
        evaluating_data (Tuple[np.ndarray, np.ndarray]): tuple containing the dataset for evaluation 
        with the same structure as training_data.

    Returns:
        float: tECV computed based on PACE.
    """
    y_mean = np.mean(training_data[1], axis=0)
    q_mean = np.mean(training_data[0], axis=0)
    cov_y = np.cov(training_data[1], rowvar=False, bias=True)

    training_data_y = training_data[1] - y_mean
    training_data_q = training_data[0] - q_mean
    cov_q_y = np.mean(training_data_q[:, :, np.newaxis] @ training_data_y[:, np.newaxis, :], axis=0)

    # Fit the PACE (linear) coefficients, then evaluate on the evaluation set
    a_pace = cov_q_y @ np.linalg.inv(cov_y)
    b_pace = q_mean - np.mean(training_data[1] @ a_pace.T, axis=0)
    pace_ce = evaluating_data[1] @ a_pace.T + b_pace
    return np.sum(np.mean((evaluating_data[0] - pace_ce) ** 2, axis=0))


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
    augmented_training_data = create_augmented_dataset(d=d, N=N)
    augmented_evaluating_data = create_augmented_dataset(d=d, N=M)
    return _pace_ecv(augmented_training_data, augmented_evaluating_data)


#--- Important sampling-based approach ---#
def importance_sampling_ecv(d: float, N_inner: int) -> float:
    """
    Compute the tECV using
    the Importance Sampling (IS) method.

    Parameters:
        d (float): Design parameter
        N_inner (int): Number of samples of the inner loop.

    Returns:
        float: tECV computed using the IS method.
    """
    # Outer dataset with size of 1
    Do = create_dataset(d=d, N=1)
    y = Do[1]

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
    posterior_mean = np.mean(q_samples * likelihood[:, np.newaxis], axis=0)
    posterior_var = np.mean(q_samples ** 2 * likelihood[:, np.newaxis], axis=0) - posterior_mean ** 2

    return np.sum(posterior_var)


#--- Functions related to observational model ---#
def linear_observation_matrix(d: float) -> np.ndarray:
    """
    Return the linear observational map's coefficient
        h(q) = Hq
    Parameters:
        d (float): Design parameter used to calculate the coefficient matrix.

    Returns:
        np.ndarray: Coefficient matrix H for the linear observational map.
    """
    H = np.identity(dim_q) / ((d - 0.5) ** 2 + 1)
    return H


def noise_free_observational_map(q: np.ndarray, d: float) -> np.ndarray:
    """
    Calculate the noise-free observational map.
    Parameters:
        q (matrix-like): Samples of the identifying random vector.
        d (float): Design parameter used in the noise-free map.
    Returns:
        matrix-like: Noise-free observational map.
    """
    H = linear_observation_matrix(d)
    return q @ H.T


def observational_map(q: np.ndarray, d: float) -> np.ndarray:
    """
    Compute the observational map by adding noise to the noise-free map.
    Parameters:
        q (matrix-like): Samples of the identifying random vector.
        d (float): Design parameter used in the noise-free map.

    Returns:
        matrix-like: Observational map obtained by adding noise to the noise-free map.
    """
    noise_free_map = noise_free_observational_map(q=q, d=d)
    epsilon = epsilon_rv.rvs(size=q.shape[0]) if len(q.shape) > 1 else epsilon_rv.rvs()
    return noise_free_map + epsilon


def create_dataset(d: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of samples by generating observations based on a noise-free observational map.

    Parameters:
        d (float): Design parameter used in the noise-free observational map.
        N (int): Number of samples to generate in the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the dataset with two elements:
            - q_samples: Matrix-like object of samples of the identifying random vector.
            - y_samples: Matrix-like object of corresponding observations.

    """
    q_samples = q_rv.rvs(size=N)
    y_samples = observational_map(q=q_samples, d=d)
    return q_samples, y_samples


def create_augmented_dataset(d: float, N: int, Na: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an augmented dataset by generating additional samples with added noise.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the base dataset.
        Na (int): Number of times to replicate the base dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the augmented dataset 
        with two elements:
            - q_samples: Matrix-like object of samples of the identifying random vector.
            - y_samples: Matrix-like object of augmented noisy observations.

    """
    q_samples_base = q_rv.rvs(size=N)
    y_samples_base = noise_free_observational_map(q=q_samples_base, d=d)
    q_samples = np.tile(q_samples_base, (Na, 1))
    y_samples = np.tile(y_samples_base, (Na, 1)) + epsilon_rv.rvs(size=N * Na)
    return q_samples, y_samples


def ref_ecv(d: float) -> float:
    """
    Reference value of the tECV

    Parameters:
        d (float): Design parameter used in the computation.

    Returns:
        float: The tECV.

    """
    H = linear_observation_matrix(d)
    observation_covariance = sigma_q ** 2 * H @ H.T + sigma_epsilon ** 2 * np.identity(dim_q)
    cross_covariance = sigma_q ** 2 * H
    K_gain_matrix = cross_covariance @ np.linalg.inv(observation_covariance)
    prior_covariance = sigma_q ** 2 * np.identity(dim_q)
    K_cross_covariance = K_gain_matrix @ cross_covariance
    K_cross_covariance_KT = K_gain_matrix @ observation_covariance @ K_gain_matrix.T
    return np.trace(prior_covariance - 2 * K_cross_covariance + K_cross_covariance_KT)


if __name__ == '__main__':

    dim_q = 10
    sigma_q = 1           # std of the identifying random variable
    sigma_epsilon = 1e-1  # std of the observational errors

    q_rv = multivariate_normal(mean=np.zeros(shape=(dim_q,)), cov=sigma_q ** 2 * np.eye(dim_q))
    epsilon_rv = multivariate_normal(mean=np.zeros(shape=(dim_q,)), cov=sigma_epsilon ** 2 * np.eye(dim_q))

    d = 0.5                                             # design parameter under test
    N = np.array([100, 500, 1000, 5000, 10000, 20000]) # PACE training/eval sample sizes
    M = N
    N_inner = [200, 500, 1000, 5000, 10000, 40000]     # IS inner-loop sample sizes
    number_runs = 100

    ecv_reference = ref_ecv(d=d)

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
    l1error_pace = np.mean(np.abs(ecv_pace - ecv_reference), axis=1)
    l1error_pace_aug = np.mean(np.abs(ecv_pace_aug - ecv_reference), axis=1)
    l1error_is = np.mean(np.abs(ecv_is - ecv_reference), axis=1)

    fig, ax = plt.subplots()
    ax.plot(N_inner, l1error_is / ecv_reference, '-x', label='IS-based')
    ax.plot(N + M, l1error_pace / ecv_reference, '-s', label='PACE-based')
    ax.plot(N + M, l1error_pace_aug / ecv_reference, '--o', label='PACE-based +  data augmentation')
    ax.plot(N + M, 2 * np.sqrt(2. / N), '--', linewidth=2, 
        label=r'$\sqrt{{2}/{N}}+\sqrt{{2}/{M}}$')  # theoretical bound

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of samples')
    ax.set_ylabel('relMAE')
    ax.legend(fontsize='10')
    ax.set_ylim([1e-4, 10])

    fig.savefig('MultiDimLinear_Gaussian_Dim_' + str(dim_q) + '.pdf')
    plt.show()
