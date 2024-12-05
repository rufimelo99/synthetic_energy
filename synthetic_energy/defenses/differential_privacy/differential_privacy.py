from enum import Enum
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class ScalerType(str, Enum):
    """
    Enumeration of available scaler types for data normalization.

    Attributes
    ----------
    STANDARD : str
        StandardScaler, which scales data to have zero mean and unit variance.
    MINMAX : str
        MinMaxScaler, which scales data to a range [0, 1].
    MAXABS : str
        MaxAbsScaler, which scales data to [-1, 1] based on the maximum absolute value.
    ROBUST : str
        RobustScaler, which scales data using median and interquartile range, robust to outliers.
    QUANTILE : str
        QuantileTransformer, which transforms data to follow a uniform or normal distribution.
    POWER : str
        PowerTransformer, which applies a power transformation to stabilize variance.
    NORMALIZER : str
        Normalizer, which scales data samples to have unit norm.
    """

    STANDARD = "standard"
    MINMAX = "minmax"
    MAXABS = "maxabs"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    NORMALIZER = "normalizer"


class DifferentialPrivacy:
    """
    A class to apply differential privacy to datasets by scaling the data and adding Laplace noise.

    Differential privacy is achieved by injecting noise into the data, ensuring that individual
    entries cannot be identified with high confidence. This class supports various data scaling
    methods for preprocessing.

    Parameters
    ----------
    epsilon : float, optional
        Privacy budget for differential privacy (default is 1.0). Smaller values result
        in more noise and greater privacy, while larger values reduce noise and increase
        data utility.
    scaler : ScalerType, optional
        The type of scaler to use for data normalization (default is ScalerType.STANDARD).
        Supported scalers are:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'maxabs': MaxAbsScaler
        - 'robust': RobustScaler
        - 'quantile': QuantileTransformer
        - 'power': PowerTransformer
        - 'normalizer': Normalizer
    """

    def __init__(
        self, epsilon: float = 1.0, scaler: ScalerType = ScalerType.STANDARD
    ) -> None:
        self.epsilon: float = epsilon
        self.scaler_type: ScalerType = scaler
        self.scaler: Optional[
            Union[
                StandardScaler,
                MinMaxScaler,
                MaxAbsScaler,
                RobustScaler,
                QuantileTransformer,
                PowerTransformer,
                Normalizer,
            ]
        ] = None

    def fit(self, X: npt.ArrayLike) -> None:
        """
        Fit the specified scaler to the input data.

        This method determines the parameters of the chosen scaler based on the input data.
        These parameters are then used for scaling in subsequent transformations.

        Parameters
        ----------
        X : array-like
            Input data used to compute the scaling parameters. Should be in the form of
            a NumPy array or similar format (e.g., Pandas DataFrame).

        Raises
        ------
        ValueError
            If an invalid scaler type is specified.
        """
        if self.scaler_type == ScalerType.STANDARD:
            self.scaler = StandardScaler()
        elif self.scaler_type == ScalerType.MINMAX:
            self.scaler = MinMaxScaler()
        elif self.scaler_type == ScalerType.MAXABS:
            self.scaler = MaxAbsScaler()
        elif self.scaler_type == ScalerType.ROBUST:
            self.scaler = RobustScaler()
        elif self.scaler_type == ScalerType.QUANTILE:
            self.scaler = QuantileTransformer()
        elif self.scaler_type == ScalerType.POWER:
            self.scaler = PowerTransformer()
        elif self.scaler_type == ScalerType.NORMALIZER:
            self.scaler = Normalizer()
        else:
            raise ValueError(f"Invalid scaler type: {self.scaler_type}")

        self.scaler.fit(X)

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Transform the input data using the fitted scaler and add Laplace noise.

        This method scales the data using the fitted scaler and injects Laplace noise
        to achieve differential privacy.

        Parameters
        ----------
        X : array-like
            Input data to be scaled and noise added. Should be in the form of a NumPy array
            or similar format.

        Returns
        -------
        np.ndarray
            Transformed data with Laplace noise added for differential privacy.

        Raises
        ------
        ValueError
            If the scaler has not been initialized. Call `fit` before using this method.
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call `fit` first.")

        X_scaled: np.ndarray = self.scaler.transform(X)
        noise: np.ndarray = np.random.laplace(0, 1 / self.epsilon, X_scaled.shape)
        return X_scaled + noise

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Fit the scaler and transform the input data in a single step.

        This method combines `fit` and `transform` for convenience.

        Parameters
        ----------
        X : array-like
            Input data to be scaled and noise added.

        Returns
        -------
        np.ndarray
            Transformed data with Laplace noise added for differential privacy.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Reverse the scaling transformation applied to the data.

        This method reverts the scaling applied by the `transform` method. However, the
        noise added for differential privacy cannot be removed, so the original data
        cannot be fully recovered.

        Parameters
        ----------
        X : array-like
            Scaled data to revert to the original scale.

        Returns
        -------
        np.ndarray
            Data rescaled to the original scale.

        Raises
        ------
        ValueError
            If the scaler has not been initialized. Call `fit` before using this method.
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call `fit` first.")

        return self.scaler.inverse_transform(X)


if __name__ == "__main__":
    X = np.random.rand(100, 10)
    dp = DifferentialPrivacy(epsilon=1.0, scaler="standard")
    X_dp = dp.fit_transform(X)
    X_inv = dp.inverse_transform(X_dp)
    print(np.allclose(X, X_inv))

    print(f" Standard Scaler: {np.mean(X - X_inv)}")
    print(f" Standard Scaler: {np.std(X - X_inv)}")
    print(f" Differential Privacy: {np.mean(X - X_dp)}")
    print(f" Differential Privacy: {np.std(X - X_dp)}")
