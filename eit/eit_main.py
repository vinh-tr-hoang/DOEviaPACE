from eit_cl import CeEIT, ConstrainedMultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.backend.set_floatx('float32')


def optimize(d_init: np.array, ce_eit: CeEIT, total_iter: int,
             training_data_size: int, M: int) -> np.array:
    """
    Function to optimize the experimental design parameters. It uses an iterative procedure where
    each iteration consists of two main steps: training an Artificial Neural Network (ANN) to approximate
    the conditional expectation of the design parameter given the observational random variable,
     and then optimizing the design parameters using the trained ANN.

    Parameters:
    d_init (np.array): Initial guess of the design parameters.
    ce_eit (CeEIT): Instance of the CeEIT class used for computations.
    total_iter (int): Total number of iterations to run the optimization procedure.
    training_data_size (int): Number of evaluations of the noise-free observational map used to train the ANN.
    M (int): Total number of evaluations of the noise-free observational map used for optimizing the design vector.
    saved_csv (str): Filename of the .csv file where results are saved at each iteration.
    """

    _d = d_init
    overall_history = []
    for i in range(total_iter):

        # defining the weighted function of the design vector based on a normal distribution
        ce_eit.d_rv = ConstrainedMultivariateNormal(mean=_d,
                                                    cov=np.eye(_d.size) * 0.2 ** 2)

        print(f'ITERATION {i} of {total_iter-1}')
        # approximating the conditional mean as an ANN using the orthogonal projection approach
        history = ce_eit.approximate_conditional_expectation(data_size=training_data_size)

        # adapted reducing learning rate used for optimizing the experimental design parameters
        if history.history['loss'][-1] > 0.3:
            lr = 1.e-1
        elif history.history['loss'][-1] > 0.2:
            lr = 0.5e-1
        else:
            lr = 0.2e-1

        if history.history['loss'][-1] < 1.02:
            # optimizing the design of experiment for 20 epoch
            # and with full-batch of size M/20
            _d, loss, lamda = ce_eit.optimize_design_parameter(initial_current=_d, epoch=20, n=M // 20,
                                                               batch_size=M // 20, lr=lr)

        # computing the total expectorated conditional variance. This is only for plotting purpose
        t_ecv = ce_eit.total_expected_conditional_variance(current=_d, ce_ANN=ce_eit.ce, nb_sample=1000)

        print(f'found design: \n  {_d} \n total expected conditional variance:  {t_ecv} \n')

        # optimization history
        _history = {'i': i, 'd': _d, 't_ecv': t_ecv}
        overall_history.append(_history)

    return _d, overall_history


def plot(ref_optimal_d: np.array, overall_history: list):
    """
    Plot the evolution of the total expected conditional variance (tECV) and
    L2-norm error of the design parameters. These are plotted with respect to the
    reference optimal design parameters over iterations.

    Parameters:
    ref_optimal_d (np.array): Reference optimal design parameters
    overall_history (list): optimization history
    """

    # optimization history of design parameters and total expected conditional variance
    _d = np.array([_['d'] for _ in overall_history])
    t_ecv = np.array([_['t_ecv'] for _ in overall_history])
    _epoch = _d.shape[0]  # Total number of epochs

    # Plot L2-norm error of the design parameters compared to the reference one over iterations
    plt.figure()
    plt.plot(range(1, _epoch + 1),
             np.linalg.norm(_d - np.repeat([ref_optimal_d], _epoch, axis=0), ord=2, axis=1), linewidth=2)
    plt.semilogy()
    plt.xlabel('iteration')
    plt.xticks(np.concatenate([[1], np.arange(0, 100, 5)]))
    plt.xlim([1, _epoch])
    plt.grid()
    plt.ylabel(r'$\Vert d - d_{{opt}}\Vert_{2}$')
    # plt.savefig('optimization_eit_d_nml_' + str(training_data_size) + '.pdf')

    # Plot the total expected conditional variance over iterations
    plt.figure()
    plt.plot(range(1, _epoch + 1), t_ecv, linewidth=2)
    plt.xlabel('iteration')
    plt.xticks(np.concatenate([[1], np.arange(0, 100, 5)]))
    plt.ylabel('tECV')
    plt.semilogy()
    plt.ylim([0.0001, 1e-2])
    plt.xlim([1, _epoch])
    plt.grid()
    # plt.savefig('optimization_eit_ecv_nml_' + str(training_data_size) + '.pdf')
    plt.show()


# Initialize the marginal standard deviation for the observational errors
epsilon_std = 10

# Instantiate CeEIT with the specified observational error standard deviation
ce_eit = CeEIT(epsilon_std=epsilon_std)

# Set the number of iterations for the main procedure
total_iter = 20

# Set the size of the training set for each iteration
training_data_size = 500
# Note: Each iteration of the main procedure requires training_data_size evaluations of the noise-free observational map.
# These evaluations are used to train the Artificial Neural Network (ANN) approximating the conditional mean.

M = 500  # Total number of evaluations of the  the noise-free observational map used for optimizing the design vector.

# An initial guess for the design parameters
d_guest = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1], dtype='float32') * 0.51

# Run the optimization process to find the optimal design of experiments
numerical_optimal_d, overall_history = optimize(d_guest, ce_eit, total_iter, training_data_size, M)
print(f'Found A-optimal design of experiments {numerical_optimal_d}')

# The reference optimal design for comparison
ref_optimal_d = np.array([1., 1., 1., -1., -1., 1., 1., -1., -1, ])
print(f'Referenced A-optimal design of experiments {ref_optimal_d}')

# Plot the evolution of the total expected conditional variance and L2-norm error of the design parameters over iterations
plot(ref_optimal_d, overall_history)
