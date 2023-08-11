import numpy as np
from scipy.stats import norm, uniform
import math
import matplotlib.pyplot as plt

# ------------------------------------- Functions related to PACE-based approach ------------------------------------- #
def _pace_ecv(training_data: [np.array, np.array], evaluating_data: [np.array, np.array]) -> float:
    """
    Compute the total expected conditional variance (ECV) using the PACE
    (Projecting Approximation of Conditional Expectation) method.

    Parameters:
        training_data (Tuple[np.array, np.array]): List containing the dataset for training, with two elements:
            - training_data[0]: Array-like object containing samples of the identifying random vector.
            - training_data[1]: Array-like object containing corresponding observations.
        evaluating_data (Tuple[np.array, np.array]): List containing the dataset for evaluation, with two elements:
            - evaluating_data[0]: Array-like object containing samples of the identifying random vector.
            - evaluating_data[1]: Array-like object containing corresponding observations.

    Returns:
        float: Total expected conditional variance computed using the PACE method.
    """

    # Compute mean of observations and identifying random vector from training dataset
    y_mean = np.mean(training_data[1])
    q_mean = np.mean(training_data[0])

    # Compute covariance matrix of observations
    cov_y = np.mean(training_data[1] * training_data[1]) - y_mean ** 2

    # Compute empirical covariance between identifying random vector and observations
    cov_q_y = np.mean((training_data[1] - y_mean) * (training_data[0] - q_mean))

    # Compute empirical PACE coefficients
    a_pace = cov_q_y * cov_y ** (-1)
    b_pace = q_mean - np.mean(a_pace * training_data[1])

    # Compute empirical PACE
    pace_ce = a_pace * evaluating_data[1] + b_pace
    return np.mean((evaluating_data[0] - pace_ce) ** 2)


def pace_ecv(d: float, N: int, M: int) -> float:
    """
    Compute the total expected conditional variance (ECV) using the PACE (Projecting Approximation of Conditional
    Expectation) method.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the training dataset.
        M (int): Number of samples in the evaluating dataset.

    Returns:
        float: Total expected conditional variance computed using the PACE method.
    """
    training_data = create_dataset(d=d, N=N)
    evaluating_data = create_dataset(d=d, N=M)
    return _pace_ecv(training_data, evaluating_data)


def data_augmented_pace_ecv(d: float, N: int, M: int) -> float:
    """
    Compute the total expected conditional variance (ECV) using the PACE (Projecting Approximation of Conditional Expectation) method with data augmentation.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the training dataset.
        M (int): Number of samples in the evaluating dataset.

    Returns:
        float: Total expected conditional variance computed using the PACE method.
    """
    training_data = create_augmented_dataset(d=d, N=N)
    evaluating_data = create_augmented_dataset(d=d, N=M)
    return _pace_ecv(training_data, evaluating_data)


# ------------------------------------------ Important sampling-based approach --------------------------------------- #
def important_sampling_ecv(d: float, N_i: int) -> float:
    """
    Compute the total expected conditional variance (ECV) using the Importance Sampling (IS) method.

    Parameters:
        d (float): Design parameter.
        N_i (int): Number of samples of the inner loop.

    Returns:
        float: Total expected conditional variance computed using the important sampling method.
    """
    # Outer dataset with size of 1
    Do = create_dataset(d=d, N=1)
    y = Do[1][0]

    # Inner dataset with size of N_i
    q_samples = q_rv.rvs(size=N_i)
    y_samples = noise_free_observational_map(q=q_samples, d=d)

    # Compute the likelihood as the density function of the difference between observed and generated samples
    likelihood = epsilon_rv.pdf(y_samples - y)

    # Normalize the likelihood by dividing it by its mean for improving numerical stability
    likelihood_mean = np.mean(likelihood)
    likelihood = likelihood / likelihood_mean

    # Compute the logarithm of the normalized likelihood with a small offset to avoid division by zero errors
    posterior_mean = np.mean(q_samples * likelihood)
    posterior_var = np.mean((q_samples - posterior_mean) ** 2 * likelihood)
    return posterior_var


# ------------------------------------- Functions related to observational model ------------------------------------- #
def linear_observation_coefficient(d: float) -> float:
    """
    Return the linear observational map's coefficient a
        h(q) = c*q
    Parameters:
        d (float): Design parameter used to calculate the coefficient matrix.

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
        q (float or np.array): Samples of the identifying random vector.
        d (float): Design parameter used in the noise-free map.

    Returns:
        float or np.ndarray: Observational map obtained by adding noise to the noise-free map.
    """
    return noise_free_observational_map(q=q, d=d) + epsilon_rv.rvs(size=q.size)


def ref_ecv(d: float) -> float:
    """
    Compute the reference expected conditional variance (ECV).

    Parameters:
        d (float): Design parameter.

    Returns:
        float: Reference expected conditional variance (ECV).
    """
    h = 1. / ((d - 0.5) ** 2 + 1)
    cov_y = h ** 2 * sigma_q ** 2 + sigma_epsilon ** 2
    cov_q_y = h * sigma_q ** 2
    k = cov_q_y * cov_y ** (-1)

    return sigma_q ** 2 - 2 * k * cov_q_y + k ** 2 * cov_y


def create_augmented_dataset(d: float, N: int, Na: int = 30) -> [np.array, np.array]:
    """
    Create an augmented dataset by generating additional samples with added noise.

    Parameters:
        d (float): Design parameter.
        N (int): Number of samples in the base dataset.
        Na (int): Number of times to replicate the base dataset.

    Returns:
        List[np.ndarray]: Augmented dataset containing samples of the identifying random vector and augmented noisy observations.
    """
    q_samples_base = q_rv.rvs(size=N)
    y_samples_base = noise_free_observational_map(q=q_samples_base, d=d)
    q_samples = np.repeat(q_samples_base, Na).reshape(-1, )
    y_samples = np.repeat(y_samples_base, Na).reshape(-1, ) + epsilon_rv.rvs(size=N * Na)
    return [q_samples, y_samples]


def create_dataset(d: float, N: int) -> [np.array, np.array]:
    """
    Create a dataset of samples by generating observations based on a noise-free observational map.

    Parameters:
        d (float): Design parameter used in the noise-free observational map.
        N (int): Number of samples to generate in the dataset.

    Returns: Tuple[np.array, np.array]: Dataset containing samples of the identifying random vector and corresponding
    observations.
    """
    q_samples = q_rv.rvs(size=N)
    y_samples = observational_map(q=q_samples, d=d)
    return [q_samples, y_samples]


def plot_expected_conditional_variance_global_approach(n=5000, number_runs=2):
    """
    Plot the statistical errors in estimating the expected conditional variance (ECV) for different methods.

    Parameters:
        n (int): Total number of samples.
        number_runs (int): Number of runs.

    Returns:
        None
    """
    # Create an evenly spaced array of design parameters values between 0 and 1
    d_array = np.linspace(0, 1, 10)

    # Initialize arrays to hold the ECV values for the PACE-based and IS-based methods for each run and each d value
    ecv_pace, ecv_is = np.zeros((number_runs, d_array.size)), np.zeros((number_runs, d_array.size))

    # Initialize an array to hold the reference ECV values for each d value
    reference_ecv = np.zeros_like(d_array)

    # For each d value, compute the reference ECV
    for i in range(d_array.size):
        reference_ecv[i] = ref_ecv(d_array[i])

    # For each run and each d value, compute the ECV using the IS-based approach
    for r in range(number_runs):
        for i in range(d_array.size):
            _ecv = important_sampling_ecv(d=d_array[i], N_i=n)

            # If the computed ECV value is NaN or greater than 4 times the reference ECV value,
            # compute a new ECV sample. This is a trick to deal with divide-by-zero errors.
            while math.isnan(_ecv) or _ecv > reference_ecv[i] * 4:
                _ecv = important_sampling_ecv(d=d_array[i], N_i=n)

            ecv_is[r, i] = _ecv

    # For each run and each d value, compute the ECV using the PACE-based approach
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

    # Standard deviation of the identifying random variable
    sigma_q = 2
    # The standard deviation of the observational errors
    sigma_epsilon = 1e-2

    # Define normal random variables of identifying parameters and observation errors
    q_rv = norm(loc=0, scale=sigma_q)
    epsilon_rv = norm(loc=0, scale=sigma_epsilon)

    # Plot the statistical errors in estimating the expected conditional variance (ECV) for different methods.
    plot_expected_conditional_variance_global_approach(n=10000, number_runs=1000)

    # Selected design parameter for testing different method
    d = 0.5
    # List of sample size in the training dataset for PACE approach
    N = np.array([100, 500, 1000, 5000, 10000, 20000])
    # List of sample size in the evaluation dataset for PACE
    M = N
    # List of sample sizes for Importance Sampling
    N_i = [200, 500, 1000, 5000, 10000, 40000]

    # Number of runs for computing statistical error
    number_runs = 100

    # Compute the reference ECV
    reference_ecv = ref_ecv(d=d)

    # Initialize arrays to store ECV values
    ecv_pace = np.zeros(shape=(len(N), number_runs))
    ecv_pace_aug = np.zeros_like(ecv_pace)
    ecv_is = np.zeros(shape=(len(N_i), number_runs))

    # Compute ECV using PACE and PACE with data augmentation
    for i in range(len(N)):
        ecv_pace[i, :] = np.array([pace_ecv(d=d, N=N[i], M=M[i]) for _ in range(number_runs)])
        ecv_pace_aug[i, :] = np.array([data_augmented_pace_ecv(d=d, N=N[i], M=M[i]) for _ in range(number_runs)])

    # Compute ECV using Importance Sampling
    for i in range(len(N_i)):
        ecv_is[i, :] = np.array([important_sampling_ecv(d=d, N_i=N_i[i]) for _ in range(number_runs)])

    # Calculate L1 error between ECV values and reference ECV
    l1error_pace = np.mean(np.abs(ecv_pace - reference_ecv), axis=1)
    l1error_pace_aug = np.mean(np.abs(ecv_pace_aug - reference_ecv), axis=1)
    l1error_is = np.mean(np.abs(ecv_is - reference_ecv), axis=1)

    # Plot relative absolute mean errors
    plt.figure('relative MSE')

    # Plot relative mean absolute errors for Importance Sampling
    plt.plot(N_i, l1error_is / reference_ecv, '-x', label='IS-based')

    # Plot relative mean absolute errors for PACE
    plt.plot(N + M, l1error_pace / reference_ecv, '-s', label='PACE-based')

    # Plot relative mean absolute errors for PACE with augmentation
    plt.plot(N + M, l1error_pace_aug / reference_ecv, '--s', label='PACE-based + data augmentation')

    # Plot theoretical bound of PACE
    plt.plot(N + M, 2 * np.sqrt(2. / N), '--k', linewidth=2,
             label=r'$\sqrt{{2}/{N}}+\sqrt{{2}/{M}}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of samples')
    plt.ylabel(r'relMAE')
    plt.legend()
    plt.ylim([1e-3, 10])

    plt.savefig('linear_gaussian_1d_sigma_eps_' + str(sigma_epsilon) + '.pdf')
    plt.show()
