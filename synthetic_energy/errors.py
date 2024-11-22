"""
Custom error classes for the `synthetic_energy` package.
"""


class SyntheticEnergyError(Exception):
    """
    Base class for all known errors in the `synthetic_energy` package.

    All errors thrown by this package should inherit from this class,
    ensuring a unified error hierarchy.
    """


class UserError(SyntheticEnergyError):
    """
    Base class for errors caused by invalid usage.

    These errors typically arise from:
    - Invalid input parameters.
    - Incorrect usage of methods on an uninitialized class.

    If this error is encountered, consult the documentation for the relevant
    class or method and verify the provided inputs.

    This class of errors corresponds to the 4xx status codes in the HTTP protocol.
    """


class InternalError(SyntheticEnergyError, RuntimeError):
    """
    Error indicating an invalid internal state.

    For documented interfaces:
        This typically points to a bug in the `synthetic_energy` package.
    For undocumented interfaces:
        This could suggest incorrect usage.

    This class of errors corresponds to the 5xx status codes in the HTTP protocol.
    """


class DataError(UserError, ValueError):
    """
    Error representing issues with training data before processing begins.

    Examples of issues include:
    - Data containing unsupported values (e.g., infinity, excessive NaNs).
    - Improper data formats such as nested structures.
    """


class ParameterError(UserError, ValueError):
    """
    Error representing problems with user-provided configurations or parameters.

    Examples of issues include:
    - Referencing columns not present in the dataset.
    - Providing invalid configuration values.
    """


class GenerationError(UserError, RuntimeError):
    """
    Error representing problems during data generation or sampling.

    Examples of issues include:
    - Rejection sampling failures.
    - Thresholds for invalid records being exceeded.
    """


# Deprecated error classes
class TooManyInvalidError(GenerationError):
    """
    Deprecated. Will be replaced by `GenerationError` in a future release.
    """

    pass


class TooFewRecordsError(DataError):
    """
    Deprecated. Will be replaced by `DataError` in a future release.
    """

    pass


class InvalidSeedError(DataError):
    """
    Deprecated. Will be replaced by `DataError` in a future release.
    """

    pass
