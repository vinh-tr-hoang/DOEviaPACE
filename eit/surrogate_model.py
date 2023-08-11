import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.keras.backend.set_floatx('float32')


class SurrogateEit:
    def __init__(self):
        """
        Initializes the SurrogateEit class.

        This class is responsible for constructing a surrogate model from Finite Element (FE) data.
        It also contains methods for reusing the created surrogate model.

        Attributes:
            design_param_dim (int): Dimension of the design vector 'd'. Here, it's set to 9.
            angle_dim (int): Dimension of the vector 'q' to be identified. Here, it's set to 2.
            MLPRegressor (tf.keras.Sequential): A multi-layer perceptron model used as the surrogate model. This model
            consists of several layers, including input, hidden, and output layers.
        """

        # dimension of the design vector (applied currents at electrodes: d)
        self.design_param_dim= 10
        # dimension of the vector to be identified (plies angles: q)
        self.angle_dim=2
        # dimension of the measurement vector (potentials at electrodes: y)
        self.potential_dim=10
        # deep feed-forward ANN used as the surrogate model
        self.MLPRegressor = tf.keras.Sequential()
        layer_options = {'kernel_initializer': 'random_normal', 'activation': 'relu', 'kernel_regularizer': None}
        # layer 1 (input)
        self.MLPRegressor.add(tf.keras.layers.Dense(100, input_shape=(self.angle_dim + self.design_param_dim -1,),
                                                    **layer_options))
        # layer 2
        self.MLPRegressor.add(tf.keras.layers.Dense(100, **layer_options))
        # layer 4 (output)
        self.MLPRegressor.add(tf.keras.layers.Dense(self.potential_dim - 1, kernel_initializer='random_normal',
                                                    activation=None,))


    def load_data(self):
        """
         Loads the processed FE data used in the EIT experiment.
         It directly reads the combined npz file containing the processed data, which includes design parameters,
        piles angles, and potentials at the electrodes.

        Returns:
            dict: A dictionary containing the processed data. This includes:
                - 'current': A nx9 matrix, where each row is a vector of design parameters 'd' sampled from a uniform distribution in [-1,1]^9.
                - 'angle': A nx2 matrix, each row of which is a two-dimensional vector of piles angles, sampled from its prior distribution.
                - 'potential': A nx9 matrix, each row of which is a vector potentials at the electrodes corresponding to the previous sample of the design parameters and angles.
                (where 'n' is the total number of samples, here, it's 86974.)
        """
        data = {}
        data = np.load ('FEdata/fem_data.npz')
        return data

    def data_visualisation (self):
        """
        Generate pairplot figures for potential, current, and plies angles  from the FE data.
        """
        # Load the FE data
        data = self.load_data()

        # Create a pairplot for potentials
        # Full  potential data:  the concatenation of the 'potential' data and the negative sum of 'potential' across each row
        potential = np.c_[np.array(data['potential']), - np.array(data['potential']).sum(axis =1)]
        df_U = pd.DataFrame(potential, columns=[f'U_{i}' for i in range(1, self.potential_dim+1)])
        g=sns.pairplot(df_U)
        g.fig.set_size_inches(8, 8)

        # Create a pairplot for design parameters (applied current)
        current = np.c_[np.array(data['current']), - np.array(data['current']).sum(axis =1)]
        df_I = pd.DataFrame(current, columns=[f'I_{i}' for i in range(1, self.design_param_dim + 1)])
        g=sns.pairplot(df_I)
        g.fig.set_size_inches(8, 8)

        # Create a pairplot for plies angles
        df_a = pd.DataFrame((data['angle']), columns=[f'angle_{i}' for i in range(1, self.angle_dim + 1)])
        g=sns.pairplot(df_a)
        g.fig.set_size_inches(5, 5)
        plt.show()

    def get_potential_scaler(self):
        """
        Prepares and returns the standard scaler for potentials data.

        It first loads the processed FE data and then fits a StandardScaler on the potentials data.
        After fitting, it saves the scaler for future use.

        Returns:
            sklearn.preprocessing.StandardScaler: A scaler fitted on the potentials data.
        """
        # Load the data using the class's load_data method
        data = self.load_data()

        # Initialize and fit a StandardScaler on the reduced potential data from the loaded data
        reduced_potential_sc = StandardScaler().fit(data['potential'])

        # Full  potential data:  the concatenation of the 'potential' data and the negative sum of 'potential' across each row
        full_potential = np.c_[np.array (data['potential']), - np.array(data['potential']).sum(axis=1)]

        # Initialize and fit a StandardScaler on the newly computed potential
        full_potential_sc = StandardScaler().fit(full_potential)

        # Save the scaler to disk for later use, with compression
        dump(full_potential_sc, 'FEdata/full_potential_scaler.bin', compress=False)
        dump(reduced_potential_sc, 'FEdata/reduced_potential_scaler.bin', compress=False)

        return full_potential_sc, reduced_potential_sc

    def training(self):
        """
        Trains the surrogate model (MLPRegressor) using the processed FE data.

        The model is trained using Adam optimizer and Mean Squared Error (MSE) loss.
        Training data is further split into training and validation subsets in a 70:30 ratio.
        The training history is returned for further analysis.

        Returns:
            Tuple[tf.keras.callbacks.History, tf.keras.Sequential]: A tuple where:
                - The first element is the training history, which includes loss and accuracy metrics over the training epochs.
                - The second element is the trained MLPRegressor model.
        """
        # Set training options: batch size, number of epochs, validation split ratio and verbosity level
        train_options = {'batch_size': 128, 'epochs': 5000, 'validation_split': 0.3, 'verbose': 1}

        # Compile the model (MLPRegressor) with Adam optimizer and mean squared error as the loss function
        self.MLPRegressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.e-4), loss='mse')

        # Load the processed FE data
        data = self.load_data()

        # Get the scaler of the potential
        _, reduced_potential_scaler= self.get_potential_scaler()

        # Scale the potential data using the obtained y scaler
        scaled_potential= reduced_potential_scaler.transform (data['potential'])

        # Combine 'angles' and 'currents' into a single matrix, which will be used as input features (X)
        X= np.hstack((data['angle'], data['current']))

        # Train the model with the prepared data and options, storing the history of the training process
        history = self.MLPRegressor.fit(X, scaled_potential, **train_options)

        return history, self.MLPRegressor


def main ():
    surrogate_eit = SurrogateEit()
    hist, model = surrogate_eit.training()
    model.save('FEdata/surrogate_model_retrain.h5')





if __name__ == '__main__':
    # Run the following commands for data visualisation
    # surrogate_eit = SurrogateEit()
    # surrogate_eit.data_visualisation()

    # Run the following commands to get the scalers
    # surrogate_eit = SurrogateEit()
    # surrogate_eit.get_potential_scaler ()

    # Run the following command to train surrogate model
    # main()

