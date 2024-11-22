from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from synthetic_energy.logger import logger


class Synthesiser(ABC):
    """
    Abstract base class for data synthesis models.

    Methods
    -------
    generate(dataframe: pd.DataFrame, features_to_gen: List[str], num_instances: int):
        Abstract method to generate synthetic data based on the input dataframe.
    """

    @abstractmethod
    def generate(
        self, dataframe: pd.DataFrame, features_to_gen: List[str], num_instances: int
    ):
        """
        Generate synthetic data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataset to base synthetic data generation on.
        features_to_gen : List[str]
            List of feature names to generate.
        num_instances : int
            Number of synthetic instances to generate.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class Generator(torch.nn.Module):
    """
    Neural network module for the generator in a GAN.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of hidden layer units.
    output_size : int
        Number of output features.

    Methods
    -------
    forward(x: torch.Tensor):
        Forward pass of the generator network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = torch.nn.Linear(input_size, hidden_size)
        self.map2 = torch.nn.Linear(hidden_size, hidden_size)
        self.map3 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Perform a forward pass through the generator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the generator.

        Returns
        -------
        torch.Tensor
            Output tensor from the generator.
        """
        x = self.activation(self.map1(x))
        x = self.activation(self.map2(x))
        return self.map3(x)


class Discriminator(torch.nn.Module):
    """
    Neural network module for the discriminator in a GAN.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of hidden layer units.
    output_size : int
        Number of output features.

    Methods
    -------
    forward(x: torch.Tensor):
        Forward pass of the discriminator network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = torch.nn.Linear(input_size, hidden_size)
        self.map2 = torch.nn.Linear(hidden_size, hidden_size)
        self.map3 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Perform a forward pass through the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the discriminator.

        Returns
        -------
        torch.Tensor
            Output tensor representing the discriminator's decision.
        """
        x = self.activation(self.map1(x))
        x = self.activation(self.map2(x))
        return torch.sigmoid(self.map3(x))


class GAN(Synthesiser):
    """
    Implementation of a Generative Adversarial Network (GAN) for data synthesis.

    Methods
    -------
    train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, train_data, num_epochs, batch_size):
        Trains the GAN model using the provided data.
    generate(dataframe: pd.DataFrame, features_to_gen: List[str], num_instances: int):
        Generates synthetic data based on the input dataframe.
    """

    def train_gan(
        self,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        criterion,
        train_data,
        num_epochs,
        batch_size,
    ):
        """
        Train the GAN model.

        Parameters
        ----------
        generator : Generator
            The generator network.
        discriminator : Discriminator
            The discriminator network.
        g_optimizer : torch.optim.Optimizer
            Optimizer for the generator.
        d_optimizer : torch.optim.Optimizer
            Optimizer for the discriminator.
        criterion : torch.nn.Module
            Loss function for training.
        train_data : torch.Tensor
            Training data.
        num_epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.

        Returns
        -------
        None
        """
        input_size = train_data.shape[1]
        for epoch in range(num_epochs):
            for i in range(0, train_data.size(0), batch_size):
                actual_batch_size = min(batch_size, train_data.size(0) - i)
                z = torch.rand((actual_batch_size, input_size))
                fake_data = generator(z)
                real_data = train_data[i : i + actual_batch_size]

                # Train the discriminator
                d_optimizer.zero_grad()
                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data)
                d_loss = -(torch.log(d_real) + torch.log(1 - d_fake)).mean()
                d_loss.backward()
                d_optimizer.step()

                # Train the generator
                z = torch.rand((actual_batch_size, input_size))
                fake_data = generator(z)
                g_optimizer.zero_grad()
                d_fake = discriminator(fake_data)
                g_loss = -torch.log(d_fake).mean()
                g_loss.backward()
                g_optimizer.step()

                if (i // batch_size) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Iteration {i // batch_size}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
                    )

    def generate(
        self, dataframe: pd.DataFrame, features_to_gen: List[str], num_instances: int
    ):
        """
        Generate synthetic data using the GAN model.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataset to base synthetic data generation on.
        features_to_gen : List[str]
            List of feature names to generate.
        num_instances : int
            Number of synthetic instances to generate.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the generated synthetic data.
        """
        # Convert non-numeric columns to labels with a label encoder
        data = dataframe.copy()
        features = data.columns

        labeler_encoders = {}
        for feature in features:
            if data[feature].dtype == "object":
                logger.info("Label encoding feature", feature=feature)
                label_encoder = LabelEncoder()
                data[feature] = label_encoder.fit_transform(data[feature])
                labeler_encoders[feature] = label_encoder

        categorical_features = []
        # Check numeric features if they are continuous or categorical
        # Check the value difference between each unique value sorted
        for feature in features:
            if feature in labeler_encoders:
                continue
            if data[feature].dtype == "int64" or data[feature].dtype == "float64":
                unique_values = data[feature].unique()
                unique_values.sort()
                diff = np.diff(unique_values)

                if len(diff) == 1 and diff[0] == 1:
                    logger.debug("Feature is continuous", feature=feature)
                else:
                    logger.debug("Feature is categorical", feature=feature)
                    data[feature] = data[feature].astype(
                        "category"
                    )  # TODO(Rui): This might be cool to utilise.
                    categorical_features.append(feature)

        # Normalize the data
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        train_data = torch.tensor(train_data.values, dtype=torch.float32)
        test_data = torch.tensor(test_data.values, dtype=torch.float32)

        # Define the hyperparameters
        input_size = train_data.shape[1]
        hidden_size = 128
        output_size = 1
        num_epochs = 2
        batch_size = 16
        learning_rate = 0.001

        # Create the generator and discriminator networks
        generator = Generator(input_size, hidden_size, input_size)
        discriminator = Discriminator(input_size, hidden_size, output_size)

        # Define the loss function and optimizer
        criterion = torch.nn.BCELoss()
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        self.train_gan(
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            criterion,
            train_data,
            num_epochs,
            batch_size,
        )

        # Generate samples
        z = torch.rand((num_instances, input_size))
        samples = generator(z).detach().numpy()

        # Inverse transform the samples
        samples = scaler.inverse_transform(samples)

        new_df = pd.DataFrame(samples, columns=data.columns)

        for feature in features:
            if feature in labeler_encoders:
                logger.info("Inverse label encoding feature", feature=feature)
                label_encoder: LabelEncoder = labeler_encoders[feature]
                new_values = np.round(samples[:, data.columns.get_loc(feature)]).astype(
                    int
                )
                res = []
                for new_value in new_values:
                    if new_value < 0:
                        res.append("Unknown")
                    else:
                        res.append(label_encoder.inverse_transform([new_value])[0])
                new_df[feature] = res

            if feature in categorical_features:
                logger.info("Round categorical feature", feature=feature)
                # TODO(Rui): Check the distributions still to make sure we do not have values that are not in the original dataset range.
                new_df[feature] = new_df[feature].apply(lambda x: round(x))

        # Crops the dataframe columns to only show the features the user wants.
        new_df = new_df[features_to_gen]

        return new_df
