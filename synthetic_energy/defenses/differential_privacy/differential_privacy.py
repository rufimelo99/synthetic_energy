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
    STANDARD = "standard"
    MINMAX = "minmax"
    MAXABS = "maxabs"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    NORMALIZER = "normalizer"


class DifferentialPrivacy:
    def __init__(
        self, epsilon: float = 1.0, scaler: ScalerType = ScalerType.STANDARD
    ) -> None:
        """
        Initialize the DifferentialPrivacy class.

        Parameters:
        - epsilon: float, privacy parameter (smaller values = more noise)
        - scaler: ScalerType, type of scaler to use ('standard', 'minmax', etc.)
        """
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
        Fit the scaler to the data.

        Parameters:
        - X: array-like, data to scale
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
        Scale the data and add Laplace noise for differential privacy.

        Parameters:
        - X: array-like, data to transform

        Returns:
        - np.ndarray: Transformed data with noise added.
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call `fit` first.")

        X_scaled: np.ndarray = self.scaler.transform(X)
        noise: np.ndarray = np.random.laplace(0, 1 / self.epsilon, X_scaled.shape)
        return X_scaled + noise

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Fit the scaler and transform the data in one step.

        Parameters:
        - X: array-like, data to fit and transform

        Returns:
        - np.ndarray: Transformed data with noise added.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: npt.ArrayLike) -> np.ndarray:
        """
        Reverse the scaling transformation (noise is not reversible).

        Parameters:
        - X: array-like, data to inverse transform

        Returns:
        - np.ndarray: Data scaled back to the original scale.
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
