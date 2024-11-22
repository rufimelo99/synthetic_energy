from enum import Enum

from pydantic import BaseModel


class MISSING_VALUES_IMPUTATION(str, Enum):
    """
    Enumeration for missing values imputation strategies.

    Options
    -------
    MEAN : str
        Replace missing values with the mean of the column.
    MEDIAN : str
        Replace missing values with the median of the column.
    MODE : str
        Replace missing values with the mode of the column.

    Notes
    -----
    Additional imputation methods such as KNN, MICE, and encoder-decoder
    may be implemented in the future.
    """

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"


class DEDUPLICATION(str, Enum):
    """
    Enumeration for deduplication strategies.

    Options
    -------
    KEEP : str
        Retain duplicate records in the dataset.
    """

    KEEP = "keep"


class NORMALISATION(str, Enum):
    """
    Enumeration for normalisation techniques.

    Options
    -------
    MIN_MAX : str
        Scale features to a fixed range [0, 1].
    Z_SCORE : str
        Standardize features by removing the mean and scaling to unit variance.
    ROBUST : str
        Scale features using statistics robust to outliers.

    Notes
    -----
    Additional normalization methods such as max-abs scaling and quantile normalization
    may be implemented in the future.
    """

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"


class CATEGORICAL_ENCODING(str, Enum):
    """
    Enumeration for categorical encoding strategies.

    Options
    -------
    LABEL : str
        Apply label encoding to categorical features.
    NONE : str
        Do not apply any encoding to categorical features.
    """

    LABEL = "label"
    NONE = "none"


class DataProcessingConfig(BaseModel):
    """
    Configuration for data preprocessing steps.

    Attributes
    ----------
    missing_imputation : MISSING_VALUES_IMPUTATION
        Strategy for imputing missing values (default: `mean`).
    deduplication : DEDUPLICATION
        Strategy for handling duplicate records (default: `keep`).
    normalisation : NORMALISATION
        Technique for feature normalization (default: `min_max`).
    categorical_encoding : CATEGORICAL_ENCODING
        Strategy for encoding categorical features (default: `label`).
    """

    missing_imputation: MISSING_VALUES_IMPUTATION = MISSING_VALUES_IMPUTATION.MEAN
    deduplication: DEDUPLICATION = DEDUPLICATION.KEEP
    normalisation: NORMALISATION = NORMALISATION.MIN_MAX
    categorical_encoding: CATEGORICAL_ENCODING = CATEGORICAL_ENCODING.LABEL


class SynthesiserConfig(BaseModel):
    """
    Configuration for the synthesizer model.

    Attributes
    ----------
    epochs : int
        Number of training epochs (default: 1).

    Notes
    -----
    Additional hyperparameters such as learning rate and batch size
    may be added in future iterations.
    """

    epochs: int = 1
