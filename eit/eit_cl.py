import numpy as np
from scipy.stats import multivariate_normal, uniform
from scipy.stats._multivariate import multivariate_normal_frozen as MultivariateNormal_frozen
import tensorflow as tf
from typing import Tuple, List
from joblib import load
tf.keras.backend.set_floatx('float32')

class CeEIT:
    """
    This class solves the A-optimal design of Electrical Impedance Tomography (EIT) experiments via a project-based
    approximation of the conditional expectation.

    Attributes:
        q_dim, epsilon_dim, design_dim (int): dimensions of vectors of identified QoIs, observational errors, and design parameter.
        d_itv (np.ndarray): interval of the design parameter
        d_rv, q_rv, epsilon_rv (rv_frozen): random variables for design parameter, prior and observational random variables.
        surrogate_eit (Keras model): ANN-based surrogate model trained using the data from the EIT's FE model.
        full_potential_scaler, reduced_potential_scaler: scalers to transform potentials.
        ce (Keras model): deep ANN used as a non-local approximation of the conditional expectation.
        train_options (dict): options for training the ANN.
        es (Keras EarlyStopping): callback for early stopping during ANN training.
        augmentation_param (int): parameter for data augmentation.

    Methods:
        noise_free_measurement_operator(q, d): Simulates a noise-free observation map.
        measurement_sim(q, d): Simulates a noisy observation map.
        approx_conditional_mean(data_size, d): Approximates the conditional expectation using an ANN map.
        cm_expected_conditional_variance(d, MC_cm, ce_ANN, data): Computes expected conditional variance.
        conditional_mean(d, y, ce_ANN): Computes the conditional expectation.
        sgd_optimize_d(n, d_init, batch_size, epoch, _lamda_max, es_epoch, lr, verbose): Performs stochastic gradient descent to find A-optimal design of experiments.
    """

    def __init__(self, epsilon_std=10.):
        # dimensions of vectors of identified QoIs, q, observational errors, xi, design parameter, d.
        self.q_dim, self.epsilon_dim, self.design_dim = 2, 10, 9
        # interval of the design parameter
        self.d_itv = np.repeat([[-1., 1.]], repeats=self.design_dim, axis=0)

        # assign a random variable to design parameter for easily generating its sample
        self.d_rv = ConstrainedMultivariateUniform(loc=self.d_itv[:, 0],
                                                   scale=self.d_itv[:, 1] - self.d_itv[:, 0])
        # prior random variable, Q
        self.q_rv = MultivariateUniform(loc=np.array([np.pi / 4.5, -np.pi / 3.5]),
                                        scale=np.array([np.pi / 3.5 - np.pi / 4.5, np.pi / 3.5 - np.pi / 4.5]))
        # observational random variable, Xi
        self.epsilon_rv = multivariate_normal(mean=np.zeros(shape=(self.epsilon_dim,)),
                                              cov=np.eye(self.epsilon_dim) * epsilon_std ** 2)

        # load surrogate model and the applied scalers on the potentials (output of the surrogate)
        # see surrogate_model.py for details on the FE data and the development of the surrogate model
        self.surrogate_eit = tf.keras.models.load_model('FEdata/surrogate_model.h5')

        self.full_potential_scaler = load('FEdata/full_potential_scaler.bin')  # for 10-dimensional vector of potentials
        self.reduced_potential_scaler = load('FEdata/reduced_potential_scaler.bin')  # for 9-dimensional vector of potentials

        # deep ANN used as a non-local approximation of the conditional expectation
        layer_options = {'kernel_initializer': 'random_normal', 'activation': 'relu', 'kernel_regularizer': None}
        tfmodel = tf.keras.Sequential()
        # layer 1 (input)
        tfmodel.add(tf.keras.layers.Dense(100, **layer_options))
        # layer 2 (input)
        tfmodel.add(tf.keras.layers.Dense(100, **layer_options))
        # layer 3 (input)
        tfmodel.add(tf.keras.layers.Dense(self.q_dim, kernel_initializer='random_normal', activation=None,
                                          kernel_regularizer='L1'))
        # decay schedule of the learning rate used for optimizing ANN
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.5e-3, decay_steps=10000,
                                                                     decay_rate=0.9)
        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        tfmodel.compile(optimizer=optimizer, loss='mse')
        self.ce = tfmodel

        self.train_options = {'batch_size': 128, 'epochs': 1000, 'validation_split': 0.5, 'shuffle': False}
        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1.e-5, patience=100,
                                                   mode='auto', baseline=None, restore_best_weights=False)
        # augmentation parameter
        self.augmentation_param = 30

    def noise_free_measurement_operator(self, angle: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Implement a noise-free observational map to predict the electrode potentials from plies angles and applied currents.
        The prediction is performed using a surrogate model trained from Finite Element (FE) simulation data.

        Parameters:
            angle (np.array): A (nx2) matrix, where each row is a vector of identifying parameters (angles of the plies).
            current (np.array): A (nx9) matrix, where each row is a vector of applied currents (design parameters).

        Returns:
            potential (np.ndarray): A (nx10) matrix where each row contains the potentials at the electrodes. The last column is the
                        negative sum of the first 9 columns, ensuring that the sum of potentials for each sample is zero.

        Note:
        The method concatenates input parameters 'q' and 'd', reshapes the resultant matrix to align with the surrogate
        model's requirements, and then performs the prediction. The predicted values are then inverse-transformed to
        maintain the original scaling of the potentials. Finally, Kirchhoff's current law is applied to ensure that the
        sum of all potentials is zero.
        """
        nb_sample = angle.shape[0]
        potential = self.surrogate_eit.predict(np.concatenate((angle, current), axis=1).reshape((nb_sample, self.q_dim
                                                                                                 + self.design_dim)),
                                               verbose=False)
        potential = self.reduced_potential_scaler.inverse_transform(potential)
        # Apply Kirchhoff's current law: sum of the potentials is zero
        potential = np.c_[potential, - potential.sum(axis=1)]

        return potential

    def noisy_measurement_operator(self, angle: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Simulates a noisy observational map that incorporates inherent system noise into the predicted electrode potentials.

        Parameters:
            angle (np.array): A (nx2) matrix, where each row is a vector of identifying parameters (angles of the plies).
            current (np.array): A (nx9) matrix, where each row is a vector of applied currents (design parameters).

        Returns:
            potential (np.ndarray): A (nx9) matrix where each row contains the noisy measurement of the potentials at the electrodes.

        Note:
        This method first uses a noise-free measurement operator to predict the electrode potentials from the input
        parameters 'q' and 'd'. It then adds a random noise factor (sampled from a predefined random variable 'epsilon_rv')
        to these noise-free measurements to simulate the inherent noise in real-world observations. The resultant
        noisy measurements are returned as the output.
        """

        potential = self.noise_free_measurement_operator(angle=angle, current=current)

        # Add random noise to the noise-free measurements
        potential = potential + self.epsilon_rv.rvs(size=potential.shape[0])

        return potential

    def approximate_conditional_expectation(self, data_size: int) -> tf.keras.callbacks.History:
        """
        Implementation of PACE, Projection-based Approximation of the Conditional Expectation, using an Artificial
        Neural Network (ANN).

        Parameters:
            data_size (int): The number of data samples to be used.

        Returns:
            history (tf.keras.callbacks.History): A record of training loss values.
        """

        # Create non-augmented training dataset
        # Get the noise free potentials corresponding to these samples
        current = self.d_rv.rvs(size=data_size)
        angle = self.q_rv.rvs(size=data_size)

        # Get the noise free potentials corresponding to these samples
        noise_free_y_samples = self.noise_free_measurement_operator(angle=angle, current=current)

        # Augment the training dataset
        angle = np.vstack([np.repeat(angle[0:data_size // 2, :], self.augmentation_param, axis=0),
                           np.repeat(angle[data_size // 2:, :], self.augmentation_param,
                                     axis=0)])

        current = np.vstack([np.repeat(current[0:data_size // 2, :], self.augmentation_param, axis=0),
                             np.repeat(current[data_size // 2:, :], self.augmentation_param,
                                       axis=0)])

        noise_free_y_samples = np.vstack(
            [np.repeat(noise_free_y_samples[0:data_size // 2, :], self.augmentation_param, axis=0),
             np.repeat(noise_free_y_samples[data_size // 2:, :], self.augmentation_param,
                       axis=0)])
        potential = noise_free_y_samples + self.epsilon_rv.rvs(size=noise_free_y_samples.shape[0])

        # Scale the potentials
        scaled_potential = self.full_potential_scaler.transform(potential)

        # Input of the map of the conditional expectation
        ce_input = np.hstack([scaled_potential, current])

        print(f"training ANN for approximating conditional expectation for {self.train_options['epochs']} epochs")
        history = self.ce.fit(ce_input, angle, **self.train_options, callbacks=[self.es], verbose=0)
        print('mse', np.mean((self.ce.predict(ce_input, verbose=0) - angle) ** 2))
        print('training ANN for approximating conditional expectation: END')

        return history
        # return self.ce, history

    def total_expected_conditional_variance(self,
                                            current: np.ndarray,
                                            nb_sample: int,
                                            ce_ANN: tf.keras.models.Model) -> float:
        """
        This method calculates the total expected conditional variance (tECV) which measures the quality of the design.
        The tECV is calculated via the law of total variance

        Args:
            current (np.array): A numpy array representing the current at each point in the design.
            nb_sample (int): Number of samples to use for calculating the ECV.
            ce_ANN (tf.keras.models.Model): The trained artificial neural network (ANN) model used for the approximation
            of the conditional expectation (CE).

        Returns:
            float: The total expected conditional variance.
        """

        # Sample angles based on the random variable distribution
        angle = self.q_rv.rvs(size=nb_sample)

        # Simulate measurements based on the given angles and current
        potential = self.noisy_measurement_operator(angle=angle, current=np.repeat([current], nb_sample, axis=0))

        # Calculate the conditional expectation
        conditional_expectation = self.conditional_expectation(potential=potential, current=current, ce_ANN=ce_ANN)

        # Calculate the expected conditional variance
        tecv = np.mean((angle - conditional_expectation) ** 2, axis=0)

        # Return the sum of the ECVs over all angles
        return sum(tecv)

    def conditional_expectation(self,
                                current: np.ndarray,
                                potential: np.ndarray,
                                ce_ANN: tf.keras.models.Model) -> np.ndarray:
        """
        Evaluate the conditional expectation of the angles given observational potentials and applied currents using
         the trained ANN.
        Parameters:
            current (np.ndarray, optional): Design parameters, a nx9 matrix where each row is a vector of design parameters. 
            potential (np.ndarray, optional): Observational random variable, a nx9 matrix where each row is a vector. 
            ce_ANN (tf.keras.models.Model): The trained ANN model for approximating the conditional expectation.

        Returns:
            np.ndarray: The computed conditional expectation.
        """

        scaled_potential = self.full_potential_scaler.transform(potential)
        current = np.repeat([current], scaled_potential.shape[0], axis=0)

        # Preparing input for the ANN approximation of the conditional expectation
        Yd = np.hstack((scaled_potential, current))

        return ce_ANN.predict(Yd, verbose=0)

    def optimize_design_parameter(self,
                                  n: int,
                                  initial_current: np.ndarray,
                                  batch_size: int = 256 * 8,
                                  epoch: int = 20,
                                  lamda_max: float = 0.5,
                                  lr: float = 1.e-3,
                                  verbose: bool = False) -> Tuple[np.ndarray, List[float], float]:
        """
        This method applies Adam optimizer to seek the A-optimal design of experiments. It optimizes the design
        parameters (applied currents) subject to the constraint that their sum is zeros and their absolute values are
        smaller than one. To implement this constraint, we optimize only the first nine applied currents and enforce
        that their sum is between -1 and 1.

        Parameters:
            n (int): Size of the data set used for training.
            initial_current (np.ndarray): Initial guess of the design parameters (vector).
            batch_size (int): Size of the batches for the SGD.
            epoch (int): Number of epochs for training.
            lamda_max (float): Maximum value for the Lagrange multiplier, used to enforce the constraint ∑d ≤ 1.
            es_epoch (int): Number of epochs for early stopping. The optimizer stops if the loss doesn't improve after es_epoch epochs.
            lr (float): Learning rate for the optimizer.
            verbose (bool): If True, intermediate results are printed during optimization.

        Returns:
            _d (np.ndarray): Optimized design parameters.
            train_loss (list): List of losses for each training epoch.
            _lamda (float): Final value of the Lagrange multiplier.

        """

        # Sampling the prior distribution
        angle = self.q_rv.rvs(size=n).astype('float32')
        angle_tf = tf.data.Dataset.from_tensor_slices(angle).shuffle(buffer_size=n).batch(batch_size)

        # Defining Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Defining mean squared error loss function
        loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')

        # Initial guest of the A-optimal design of experiment
        current = tf.Variable(initial_current, dtype='float32')

        # Defining Lagrange multiplier used for enforce the constrain abs(sum(d[0:-1])) <= 1
        lamda = tf.Variable(0., dtype='float32')

        # Initiate dummy value for the init loss
        # _old_loss = 9999.

        # Initiate counter for epoch with unimproved losses
        # _un_improved_counter = 0.

        # Holder of training losses for each epoch
        train_loss = []

        # Start optimization process
        for _epoch in range(epoch):
            epoch_loss = 0.
            for step, batch_current in enumerate(angle_tf):
                batch_loss = self._optimize_batch(batch_current, current, lamda, lamda_max, loss_fn, optimizer)
                epoch_loss += batch_loss

            train_loss.append(epoch_loss)

            if _epoch % 2 == 0 and verbose:
                print("epoch:", _epoch, "Loss:", float(epoch_loss), 'Lagrange multiplier', lamda)
                print('design:', current)

        return current.numpy(), train_loss, lamda.numpy()

    def _optimize_batch(self,
                        batch_current: tf.Tensor,
                        current: tf.Variable,
                        lamda: tf.Variable,
                        lamda_max: float,
                        loss_fn: tf.keras.losses.Loss,
                        optimizer: tf.keras.optimizers.Optimizer) -> float:

        batch_size = batch_current.shape[0] * self.augmentation_param
        with tf.GradientTape() as tape:

            # Collection of variables to be optimized, i.e., vector of design parameters
            # and Lagrange multiplier
            vars = [current, lamda]
            tape.watch(vars)

            # Input (plies angles and currents) of the surrogate model
            sg_input = tf.concat([batch_current, tf.repeat([current], batch_current.shape[0], axis=0)], axis=1)

            # Output (potentials) of the surrogate model
            reduced_noise_free_potential = self.surrogate_eit(sg_input) * self.reduced_potential_scaler.scale_ + \
                                           self.reduced_potential_scaler.mean_
            noise_free_potential = tf.concat([reduced_noise_free_potential,
                                              - tf.reshape(
                                                  tf.math.reduce_sum(reduced_noise_free_potential, axis=1),
                                                  [reduced_noise_free_potential.shape[0], 1])],
                                             axis=1)
            # Augment datasets
            aug_potential = tf.repeat(noise_free_potential, self.augmentation_param,
                                      axis=0) + self.epsilon_rv.rvs(size=batch_size)
            aug_current = tf.repeat(batch_current, self.augmentation_param, axis=0)

            # Inputs of the conditional expectation's ANN
            potential_scaled = (aug_potential - self.full_potential_scaler.mean_) / self.full_potential_scaler.scale_
            ce_input = tf.concat([potential_scaled, tf.repeat([current], batch_size, axis=0)], axis=1)

            # Conditional expectation
            ce = self.ce(ce_input)

            # Batch MSE
            batch_loss = loss_fn(y_true=aug_current, y_pred=ce)

            # Total loss with Lagrange penalty to enforce the constraint abs(sum(current[0:-1])) <= 1
            loss = batch_loss + lamda * tf.math.abs(tf.math.reduce_sum(current) - 1)

            # Get gradients of the loss
        gradients = tape.gradient(loss, vars)  # [-self.epsilon_dim:]

        # Update the design parameters
        optimizer.apply_gradients(zip([gradients[0]], [vars[0]]))

        # Update the Lagrange multiplier
        optimizer.apply_gradients(zip([-gradients[1]], [vars[1]]))

        # Apply admissible condition on _current
        # current = tf.Variable(self.appy_bound(current.numpy()))

        # Apply admissible bound [-1, 1] on currents
        current.assign(self.appy_bound(current.numpy()))

        # Update Lagrange multiplier based on the current sum
        current_sum = np.abs(np.sum(current.numpy()))
        if lamda.numpy() <= 0. or current_sum - 1. <= 0:
            lamda.assign(0.)

        if lamda.numpy() >= lamda_max:
            lamda.assign(lamda_max)

        return loss.numpy()

    def appy_bound(self, d: np.array) -> np.array:
        """
         Applying bounds [-1, 1] on components of vector d
        """
        eps = -1e-4

        # If there are any values in d that are less than the lower bound, set them to the lower bound
        _i = np.array(np.where(d[:] < self.d_itv[:, 0] - eps)[:])
        if _i.size > 0:
            d[_i] = self.d_itv[_i, 0] - eps

        # If there are any values in d that are greater than the upper bound, set them to the upper bound
        _i = np.array(np.where(d > self.d_itv[:, 1] + eps)[:])
        if _i.size > 0:
            d[_i] = self.d_itv[_i, 1] + eps

        return d


# ---------------------------------- Auxiliary tools for sampling random vectors ------------------------------------- #
class MultivariateUniform:
    """
    Class for multivariate independent uniform distribution. Each variable in the distribution is independent and
    follows a univariate uniform distribution. 

    Attributes:
        loc (np.ndarray): Location parameters for the uniform distributions.
        scale (np.ndarray): Scale parameters for the uniform distributions.
    """

    def __init__(self, loc: np.ndarray, scale: np.ndarray):
        """
        Args:
            loc (np.ndarray): Location parameters for the uniform distributions.
            scale (np.ndarray): Scale parameters for the uniform distributions.
        """
        assert len(loc) == len(scale), "loc and scale must have the same length"
        self.scale = scale
        self.loc = loc
        self._dim = np.size(loc)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the multivariate uniform distribution.

        Args:
            size (int): Number of samples to draw. Default is 1.

        Returns:
            np.ndarray: Random samples. Each row corresponds to a sample.
        """
        assert size > 0, "size must be a positive integer"
        x = uniform.rvs(size=size * self._dim, loc=0, scale=1).reshape((size, self._dim))
        loc = np.repeat([self.loc], repeats=size, axis=0)
        x = np.matmul(x, np.diag(self.scale)) + loc
        return x


class ConstrainedMultivariateUniform(MultivariateUniform):
    """
    Class for multivariate independent uniform distribution with a constraint. This class extends the
    MultivariateUniform class with a constraint that the sum of the random variable vector must be less than 1.

    Attributes:
        loc (np.ndarray): Location parameters for the uniform distributions.
        scale (np.ndarray): Scale parameters for the uniform distributions.
    """

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the multivariate uniform distribution with the sum of the random variable vector
        being less than 1.

        Args:
            size (int): Number of samples to draw. Default is 1.

        Returns:
            np.ndarray: Random samples. Each row corresponds to a sample.
        """
        assert size > 0, "size must be a positive integer"

        samples = super().rvs(size=size)

        # Ensure the constraint is satisfied for each sample
        for i in range(size):
            while np.abs(np.sum(samples[i, :])) > 1.:
                samples[i, :] = super().rvs(size=1)

        return samples


class ConstrainedMultivariateNormal(MultivariateNormal_frozen):
    """
    Class for multivariate normal distribution with a constraint. This class extends the 
    MultivariateNormal class with a constraint that the sum of the random variable vector must be less than 1.

    Inherits all the methods and properties of the MultivariateNormal class.
    """

    def rvs(self, size: int = 1):
        """
        Draw random samples from the multivariate normal distribution with the sum of the random variable vector 
        being less than 1.

        Args:
            size (int): Number of samples to draw. Default is 1.

        Returns:
            np.ndarray: Random samples. Each row corresponds to a sample.
        """
        assert size > 0, "size must be a positive integer"

        samples = super().rvs(size=size)

        # Ensure the constraint is satisfied for each sample
        for i in range(size):
            while np.abs(np.sum(samples[i, :])) > 1.:
                samples[i, :] = super().rvs(size=1)

        return samples
