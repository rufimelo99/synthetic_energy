"""
This module contains the implementation of the Kullback-Leibler divergence
metric for comparing two probability distributions.
"""

from abc import ABC
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance

from synthetic_energy.logger import logger


class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.
    """

    def _distance(self, p: List[List[float]], q: List[List[float]]) -> float:
        """
        Compute the distance between two probability distributions.

        Parameters
        ----------
        p : List[List[float]]
            The first probability distribution.
        q : List[List[float]]
            The second probability distribution.

        Returns
        -------
        float
            The distance between the two probability distributions.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError

    def distance(
        self, df_real: pd.DataFrame, df_generated: pd.DataFrame
    ) -> Optional[float]:
        """
        Compute the distance between two sets of probability distributions.

        Parameters
        ----------
        df_real : pd.DataFrame
            The first set of probability distributions.
        df_generated : pd.DataFrame
            The second set of probability distributions.

        Returns
        -------
        Optional[float]
            The distance between the two sets of probability distributions, or None if not computable.

        Notes
        -----
        Converts the input dataframes into probability distributions before computing the distance.

        See Also
        --------
        convert_to_probability_distribution : Converts dataframes to probability distributions.
        _distance : Computes the distance between two probability distributions.
        """
        logger.debug("Converting real dataframe to probability distribution.")
        real_prob_dist, generated_prob_dist = self.convert_to_probability_distribution(
            df_real, df_generated
        )
        return self._distance(real_prob_dist, generated_prob_dist)

    def convert_to_probability_distribution(
        self, df_real: pd.DataFrame, df_generated: pd.DataFrame
    ) -> List[List[float]]:
        """
        Convert dataframes to probability distributions.

        Parameters
        ----------
        df_real : pd.DataFrame
            The dataframe representing the real data.
        df_generated : pd.DataFrame
            The dataframe representing the generated data.

        Returns
        -------
        List[List[float]]
            A list of two probability distributions: the first corresponds to the real data,
            and the second corresponds to the generated data.

        Notes
        -----
        The method processes the input dataframes to ensure they represent valid probability distributions.
        """
        # User kernel density estimation to estimate the probability distribution of the data.
        real_distributions = []
        generated_distributions = []
        # Applies smotthing by adding a small epsilon to avoid division by zero.
        epsilon = 1e-10
        for column in df_generated.columns:
            if pd.api.types.is_datetime64_any_dtype(df_real[column]):
                logger.error(
                    "Datetime columns detected in real dataframe... Still unsure how to handle this."
                )
                continue

            # Check if a column is categorical.
            if df_generated.dtypes[column] == "object":
                real_dist = df_real[column].value_counts(normalize=True).sort_index()
                generated_dist = (
                    df_generated[column].value_counts(normalize=True).sort_index()
                )
                aligned_dist = pd.concat([real_dist, generated_dist], axis=1).fillna(0)

                # Adds epsilon to avoid division by zero.
                aligned_dist += epsilon

                real_distributions.append(aligned_dist.iloc[:, 0].values)
                generated_distributions.append(aligned_dist.iloc[:, 1].values)
            elif (
                df_generated.dtypes[column] == "float64"
                or df_generated.dtypes[column] == "int64"
            ):
                min_value = min(df_real[column].min(), df_generated[column].min())
                max_value = max(df_real[column].max(), df_generated[column].max())

                # Create 10 bins between min and max values.
                bins = np.linspace(min_value, max_value, 10)

                # Create the probabilities for each bin.
                real_hist, _ = np.histogram(df_real[column], bins=bins)
                real_prob = real_hist / real_hist.sum()

                generated_hist, _ = np.histogram(df_generated[column], bins=bins)
                generated_prob = generated_hist / generated_hist.sum()

                # Adds epsilon to avoid division by zero.
                real_prob += epsilon
                generated_prob += epsilon

                real_distributions.append(real_prob)
                generated_distributions.append(generated_prob)
            else:
                logger.warning(
                    "Unsupported data type for column",
                    column=column,
                    dtype=df_generated.dtypes[column],
                )
                continue

        return real_distributions, generated_distributions

    def __call__(self, df_real: pd.DataFrame, df_generated: pd.DataFrame) -> float:
        """
        Compute the distance between two sets of probability distributions.

        Parameters
        ----------
        df_real : pd.DataFrame
            The dataframe representing the real data.
        df_generated : pd.DataFrame
            The dataframe representing the generated data.

        Returns
        -------
        float
            The distance between the two sets of probability distributions.

        Notes
        -----
        This method serves as a callable interface to compute the distance, internally
        invoking the `distance` method.
        """
        return self.distance(df_real, df_generated)


class WassersteinDistance(DistanceMetric):
    """
    Implementation of the Wasserstein distance metric.

    The Wasserstein distance measures the cost of transforming one probability
    distribution into another. It is widely used in comparing distributions
    in statistical and machine learning tasks.
    """

    def _distance(self, p: List[List[float]], q: List[List[float]]) -> float:
        """
        Compute the Wasserstein distance between two probability distributions.

        Parameters
        ----------
        p : List[List[float]]
            The first probability distribution, represented as a list of lists.
        q : List[List[float]]
            The second probability distribution, represented as a list of lists.

        Returns
        -------
        float
            The Wasserstein distance between the two probability distributions.

        Notes
        -----
        The method calculates the Wasserstein distance feature by feature and sums
        the results to compute the overall distance.

        See Also
        --------
        wasserstein_distance : Computes the Wasserstein distance for a single feature.
        """
        wasserstein_distance_value = 0
        for feature in range(len(p)):
            wasserstein_distance_value += wasserstein_distance(p[feature], q[feature])
        logger.debug("Wasserstein distance", value=wasserstein_distance_value)
        return wasserstein_distance_value


class KLDivergence(DistanceMetric):
    """
    Implementation of the Kullback-Leibler (KL) divergence metric.

    KL divergence is a measure of how one probability distribution diverges
    from a second, expected probability distribution. It is often used to
    quantify information loss in approximations.
    """

    def _distance(self, p: List[List[float]], q: List[List[float]]) -> float:
        """
        Compute the Kullback-Leibler divergence between two probability distributions.

        Parameters
        ----------
        p : List[List[float]]
            The first probability distribution, represented as a list of lists.
        q : List[List[float]]
            The second probability distribution, represented as a list of lists.

        Returns
        -------
        float
            The Kullback-Leibler divergence between the two probability distributions.

        Notes
        -----
        The KL divergence is calculated using the formula:
        .. math::
            D_{KL}(P || Q) = \sum P(i) \log\left(\frac{P(i)}{Q(i)}\right)

        This method computes the KL divergence for each feature and sums the results
        to obtain the overall divergence.

        See Also
        --------
        scipy.stats.entropy : Computes the KL divergence for individual distributions.
        """
        return np.sum(entropy(p_i, q_i) for p_i, q_i in zip(p, q))


class PopulationStabilityIndex(KLDivergence):
    """
    Implementation of the Population Stability Index (PSI).

    PSI is a symmetric metric that compares two probability distributions.
    Unlike KL-Divergence, which is asymmetric, PSI adds the divergences in both directions
    to ensure symmetry. This property makes PSI more robust for certain use cases, such as
    monitoring model stability over time.

    Formula
    -------
    PSI = D_KL(P || Q) + D_KL(Q || P)
    """

    def _distance(self, p: List[List[float]], q: List[List[float]]) -> float:
        """
        Compute the Population Stability Index (PSI) between two probability distributions.

        Parameters
        ----------
        p : List[List[float]]
            The first probability distribution, represented as a list of lists.
        q : List[List[float]]
            The second probability distribution, represented as a list of lists.

        Returns
        -------
        float
            The Population Stability Index, a symmetric measure of divergence between the
            two probability distributions.

        Notes
        -----
        PSI is computed as the sum of the KL divergences in both directions:
        .. math::
            PSI = D_{KL}(P || Q) + D_{KL}(Q || P)

        See Also
        --------
        KLDivergence._distance : Computes the KL divergence between two distributions.
        """
        return super()._distance(p, q) + super()._distance(q, p)
