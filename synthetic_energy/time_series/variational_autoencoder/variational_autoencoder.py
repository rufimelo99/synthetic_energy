"""
Variational Autoencoder (VAE) implementation for time series data. A VAE is a generative model
that learns to encode data into a latent representation and decode from the latent space
to reconstruct the input data. This implementation includes methods for sampling, generating, 
encoding, and decoding.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) for modeling time series data.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    hidden_dim : int
        Number of units in the hidden layers.
    latent_dim : int
        Dimensionality of the latent representation.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, log_var):
        """
        Samples from a Gaussian distribution using the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent distribution.
        log_var : torch.Tensor
            The log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent variables.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Encodes input data, samples from the latent space, and reconstructs the input.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        tuple
            Reconstructed data, mean, and log variance of the latent representation.
        """
        mu_log_var = self.encoder(x)
        mu = mu_log_var[:, : self.latent_dim]
        log_var = mu_log_var[:, self.latent_dim :]
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        """
        Computes the VAE loss, including reconstruction loss and KL divergence.

        Parameters
        ----------
        x : torch.Tensor
            Original input data.
        x_hat : torch.Tensor
            Reconstructed data.
        mu : torch.Tensor
            Mean of the latent distribution.
        log_var : torch.Tensor
            Log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Total loss combining reconstruction loss and KL divergence.
        """
        reconst_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconst_loss + kl_div

    def sample(self, n_samples):
        """
        Samples data points from the latent space.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        z = torch.randn(n_samples, self.latent_dim)
        return self.decoder(z)

    def generate(self, x):
        """
        Generates new data by encoding input and decoding from the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        torch.Tensor
            Generated data.
        """
        mu_log_var = self.encoder(x)
        mu = mu_log_var[:, : self.latent_dim]
        log_var = mu_log_var[:, self.latent_dim :]
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)

    def encode(self, x):
        """
        Encodes input data into latent variables.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        tuple
            Mean and log variance of the latent variables.
        """
        mu_log_var = self.encoder(x)
        mu = mu_log_var[:, : self.latent_dim]
        log_var = mu_log_var[:, self.latent_dim :]
        return mu, log_var

    def decode(self, z):
        """
        Decodes latent variables into reconstructed data.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables.

        Returns
        -------
        torch.Tensor
            Reconstructed data.
        """
        return self.decoder(z)

    def save(self, path):
        """
        Saves the model's state dictionary to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads the model's state dictionary from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model.
        """
        self.load_state_dict(torch.load(path))

    @property
    def latent_dim(self):
        """
        Returns the dimensionality of the latent space.

        Returns
        -------
        int
            Latent space dimensionality.
        """
        return self.decoder[0].in_features


# Example usage
if __name__ == "__main__":
    vae = VAE(10, 128, 2)
    x = torch.randn(32, 10)
    x_hat, mu, log_var = vae(x)
    loss = vae.loss(x, x_hat, mu, log_var)
    print(loss)
    n_samples = 32
    samples = vae.sample(n_samples)
    print(samples)
    generated = vae.generate(x)
    print(generated)
    mu, log_var = vae.encode(x)
    print(mu, log_var)
    z = torch.randn(n_samples, 2)
    decoded = vae.decode(z)
    print(decoded)
    vae.save("vae.pth")
    vae.load("vae.pth")
