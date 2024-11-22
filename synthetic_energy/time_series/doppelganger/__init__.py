# Greatly inspired by https://github.com/gretelai/gretel-synthetics .
from synthetic_energy.time_series.doppelganger.config import (
    DfStyle,
    DGANConfig,
    Normalization,
    OutputType,
)
from synthetic_energy.time_series.doppelganger.doppelganger import (
    DGAN,
    _add_generation_flag,
    _DataFrameConverter,
    _discrete_cols_to_int,
    _LongDataFrameConverter,
    _WideDataFrameConverter,
    find_max_consecutive_nans,
    nan_linear_interpolation,
    validation_check,
)
from synthetic_energy.time_series.doppelganger.structures import ProgressInfo
from synthetic_energy.time_series.doppelganger.torch_modules import (
    Discriminator,
    Generator,
    Merger,
    OutputDecoder,
    SelectLastCell,
)
from synthetic_energy.time_series.doppelganger.transformations import (
    BinaryEncodedOutput,
    ContinuousOutput,
    OneHotEncodedOutput,
    Output,
    _grouped_min_and_max,
    create_output,
    create_outputs_from_data,
    nan_linear_interpolation,
    rescale,
    rescale_inverse,
    transform_attributes,
    transform_features,
)

__all__ = [
    "DGANConfig",
    "Normalization",
    "OutputType",
    "DfStyle",
    "DGAN",
    "_DataFrameConverter",
    "_WideDataFrameConverter",
    "_discrete_cols_to_int",
    "_add_generation_flag",
    "_LongDataFrameConverter",
    "find_max_consecutive_nans",
    "validation_check",
    "nan_linear_interpolation",
    "ProgressInfo",
    "Merger",
    "OutputDecoder",
    "SelectLastCell",
    "Generator",
    "Discriminator",
    "Output",
    "OneHotEncodedOutput",
    "BinaryEncodedOutput",
    "ContinuousOutput",
    "create_outputs_from_data",
    "create_output",
    "rescale",
    "transform_attributes",
    "rescale_inverse",
    "_grouped_min_and_max",
    "transform_features",
    "nan_linear_interpolation",
]
