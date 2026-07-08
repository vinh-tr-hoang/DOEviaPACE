import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # required to load the legacy Keras-2 .h5 surrogate model
import numpy as np
from scipy.stats import multivariate_normal, uniform
import tensorflow as tf
from typing import Tuple, List
from joblib import load
tf.keras.backend.set_floatx('float32')

class CeEIT:
    """
    This class solves the A-optimal design of Electrical Impedance Tomography (EIT) experiments via a projection-based
    approximation of the conditional expectation (PACE).

    Attributes:
        q_dim, epsilon_dim, design_dim (int): dimensions of vectors of identified QoIs, observational errors, and design parameter.
        d_itv (np.ndarray): interval of the design parameter
        d_rv, q_rv, epsilon_rv: random variables for design parameter, prior and observational random variables.
        surrogate_eit (Keras model): ANN-based surrogate model trained using the data from the EIT's FE model.
        full_potential_scaler, reduced_potential_scaler: scalers to transform potentials.
        ce (Keras model): deep ANN used as a non-local approximation of the conditional expectation.
        train_options (dict): options for training the ANN.
        es (Keras EarlyStopping): callback for early stopping during ANN training.
        augmentation_param (int): parameter for data augmentation.
    """

    def __init__(self, epsilon_std=10.):
        # dimensions of vectors of identified QoIs (q), observational errors (epsilon), design parameter (d).
        self.q_dim, self.epsilon_dim, self.design_dim = 2, 10, 9
        # interval of the design parameter
        self.d_itv = np.repeat([[-1., 1.]], repeats=self.design_dim, axis=0)

        # assign a random variable to design parameter for easily generating its sample
        self.d_rv = ConstrainedMultivariateUniform(loc=self.d_itv[:, 0],
                                                   scale=self.d_itv[:, 1] - self.d_itv[:, 0])
        # prior random variable, Q
        self.q_rv = MultivariateUniform(loc=np.array([np.pi / 4.5, -np.pi / 3.5]),
                                        scale=np.array([np.pi / 3.5 - np.pi / 4.5, np.pi / 3.5 - np.pi / 4.5]))
        # observational random variable, epsilon
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
        tfmodel.add(tf.keras.layers.Dense(100, **layer_options))
        tfmodel.add(tf.keras.layers.Dense(100, **layer_options))
        tfmodel.add(tf.keras.layers.Dense(self.q_dim, kernel_initializer='random_normal', activation=None,
                                          kernel_regularizer='L1'))
        # decay schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.5e-3, decay_steps=10000,
                                                                     decay_rate=0.9)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        tfmodel.compile(optimizer=optimizer, loss='mse')
        self.ce = tfmodel

        self.train_options = {'batch_size': 128, 'epochs': 1000, 'validation_split': 0.5, 'shuffle': False}
        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1.e-5, patience=100,
                                                   mode='auto', baseline=None, restore_best_weights=False)
        self.augmentation_param = 30

    def noise_free_measurement_operator(self, angle: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Implement a noise-free observational map to predict the electrode potentials from plies angles and applied currents.
        The prediction is performed using a surrogate model trained from Finite Element (FE) simulation data.

        Parameters:
            angle (np.ndarray): A (nx2) matrix, where each row is a vector of identifying parameters (angles of the plies).
            current (np.ndarray): A (nx9) matrix, where each row is a vector of applied currents (design parameters).

        Returns:
            potential (np.ndarray): A (nx10) matrix where each row contains the potentials at the electrodes. The last column is the
                        negative sum of the first 9 columns, ensuring that the sum of potentials for each sample is zero.

        Note:
        The method concatenates the 'angle' and 'current' inputs, feeds them to the surrogate model to predict the
        potentials, inverse-transforms the prediction to restore the original scaling, and finally fixes the 10th
        electrode potential as the negative sum of the other 9 so that the potentials sum to zero.
        """
        sg_input = np.concatenate((angle, current), axis=1)
        potential = self.surrogate_eit(sg_input).numpy()
        potential = self.reduced_potential_scaler.inverse_transform(potential)
        # Fix the 10th potential as the negative sum of the other 9 (potentials sum to zero)
        potential = np.c_[potential, - potential.sum(axis=1)]

        return potential

    def noisy_measurement_operator(self, angle: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Simulate a noisy observational map that adds noise to the predicted electrode potentials.

        Parameters:
            angle (np.ndarray): A (nx2) matrix, where each row is a vector of identifying parameters (angles of the plies).
            current (np.ndarray): A (nx9) matrix, where each row is a vector of applied currents (design parameters).

        Returns:
            potential (np.ndarray): A (nx10) matrix where each row contains the noisy measurement of the potentials at the electrodes.

        Note:
        Adds noise sampled from epsilon_rv to the noise-free measurement operator's output.
        """

        potential = self.noise_free_measurement_operator(angle=angle, current=current)

        # Add random noise to the noise-free measurements
        potential = potential + self.epsilon_rv.rvs(size=potential.shape[0])

        return potential

    def approximate_conditional_expectation(self, data_size: int) -> tf.keras.callbacks.History:
        """
        Train the ANN that approximates the conditional expectation (PACE).

        Parameters:
            data_size (int): The number of data samples to be used.

        Returns:
            history (tf.keras.callbacks.History): A record of training loss values.
        """

        # Create non-augmented training dataset
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

        # Build the conditional-expectation ANN input
        ce_input = np.hstack([scaled_potential, current])

        print(f"training ANN for approximating conditional expectation for {self.train_options['epochs']} epochs")
        history = self.ce.fit(ce_input, angle, **self.train_options, callbacks=[self.es], verbose=0)
        print('mse', np.mean((self.ce.predict(ce_input, verbose=0) - angle) ** 2))
        print('training ANN for approximating conditional expectation: END')

        return history

    def total_expected_conditional_variance(self,
                                            current: np.ndarray,
                                            nb_sample: int,
                                            ce_ANN: tf.keras.models.Model) -> float:
        """
        Calculate the total expected conditional variance (tECV), which measures the quality of the design.

        Args:
            current (np.ndarray): The design vector (applied currents).
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
            current (np.ndarray): A single design vector of 9 applied currents, applied to every row of `potential`.
            potential (np.ndarray): Observational random variable, a nx10 matrix where each row is a vector. 
            ce_ANN (tf.keras.models.Model): The trained ANN model for approximating the conditional expectation.

        Returns:
            np.ndarray: The computed conditional expectation.
        """

        scaled_potential = self.full_potential_scaler.transform(potential)
        current = np.repeat([current], scaled_potential.shape[0], axis=0)

        # Build the conditional-expectation ANN input
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
        Apply the Adam optimizer to seek the A-optimal design of experiments. It optimizes the design parameters
        (applied currents) subject to the constraint that the absolute value of their sum is at most one and their
        individual absolute values are smaller than one (enforced on the nine independent currents).

        Parameters:
            n (int): Size of the data set used for training.
            initial_current (np.ndarray): Initial guess of the design parameters (vector).
            batch_size (int): Size of the batches for the SGD.
            epoch (int): Number of epochs for training.
            lamda_max (float): Maximum value for the Lagrange multiplier, used to enforce the constraint |∑d| ≤ 1.
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

        # Initial guess of the A-optimal design of experiment
        current = tf.Variable(initial_current, dtype='float32')

        # Defining Lagrange multiplier used to enforce the constraint abs(sum(d)) <= 1
        lamda = tf.Variable(0., dtype='float32')

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
            trainables = [current, lamda]
            tape.watch(trainables)

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

            # Build the conditional-expectation ANN input
            potential_scaled = (aug_potential - self.full_potential_scaler.mean_) / self.full_potential_scaler.scale_
            ce_input = tf.concat([potential_scaled, tf.repeat([current], batch_size, axis=0)], axis=1)

            # Conditional expectation
            ce = self.ce(ce_input)

            # Batch MSE
            batch_loss = loss_fn(y_true=aug_current, y_pred=ce)

            # Total loss with Lagrange penalty to enforce the constraint abs(sum(current)) <= 1
            loss = batch_loss + lamda * tf.math.abs(tf.math.reduce_sum(current) - 1)

        # Get gradients of the loss
        gradients = tape.gradient(loss, trainables)

        # Update the design parameters and Lagrange multiplier together
        optimizer.apply_gradients(zip([gradients[0], -gradients[1]], [trainables[0], trainables[1]]))

        # Apply admissible bound [-1, 1] on currents
        current.assign(self.apply_bound(current.numpy()))

        # Update Lagrange multiplier based on the current sum
        current_sum = np.abs(np.sum(current.numpy()))
        if lamda.numpy() <= 0. or current_sum - 1. <= 0:
            lamda.assign(0.)

        if lamda.numpy() >= lamda_max:
            lamda.assign(lamda_max)

        return loss.numpy()

    def apply_bound(self, d: np.ndarray) -> np.ndarray:
        """
        Clip the components of vector d to the admissible range [-1, 1].
        """
        eps = -1e-4
        return np.clip(d, self.d_itv[:, 0] - eps, self.d_itv[:, 1] + eps).astype(d.dtype)


# ----------------- Auxiliary tools for sampling random vectors ------------------------- #
class MultivariateUniform:
    """
    Class for multivariate independent uniform distribution. 
    Each variable in the distribution is independent and follows a univariate uniform distribution. 

    Attributes:
        loc (np.ndarray): Location parameters for the uniform distributions.
        scale (np.ndarray): Scale parameters for the uniform distributions.
    """

    def __init__(self, loc: np.ndarray, scale: np.ndarray):
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
        x = x * self.scale + loc
        return x


class ConstrainedMultivariateUniform(MultivariateUniform):
    """
    Class for multivariate independent uniform distribution with a constraint. 
    This class extends the MultivariateUniform class with a constraint that the sum 
    of the random variable vector must have absolute value at most 1.

    Attributes:
        loc (np.ndarray): Location parameters for the uniform distributions.
        scale (np.ndarray): Scale parameters for the uniform distributions.
    """

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the multivariate uniform distribution with the sum of the random variable vector
        having absolute value at most 1.

        Args:
            size (int): Number of samples to draw. Default is 1.

        Returns:
            np.ndarray: Random samples. Each row corresponds to a sample.
        """
        assert size > 0, "size must be a positive integer"

        samples = super().rvs(size=size)

        # Ensure the constraint is satisfied for each sample (bounded retries)
        for i in range(size):
            attempts = 0
            while np.abs(np.sum(samples[i, :])) > 1. and attempts < 1000:
                samples[i, :] = super().rvs(size=1)
                attempts += 1

        return samples


class ConstrainedMultivariateNormal:
    """
    Multivariate normal distribution constrained so that the sum of each sampled
    vector has absolute value at most 1.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self._rv = multivariate_normal(mean=mean, cov=cov)

    def rvs(self, size: int = 1):
        """
        Draw random samples whose sum has absolute value at most 1.

        Args:
            size (int): Number of samples to draw. Default is 1.

        Returns:
            np.ndarray: Random samples. Each row corresponds to a sample.
        """
        assert size > 0, "size must be a positive integer"

        samples = self._rv.rvs(size=size)

        # Ensure the constraint is satisfied for each sample (bounded retries)
        for i in range(size):
            attempts = 0
            while np.abs(np.sum(samples[i, :])) > 1. and attempts < 1000:
                samples[i, :] = self._rv.rvs(size=1)
                attempts += 1

        return samples
