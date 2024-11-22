"""
Unified types for Gretel Synthetics.

This module defines common type aliases used throughout the Gretel Synthetics package
to provide flexibility in handling various data structures.
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import pandas as pd

# Type aliases for common data representations
DFLike = Union[pd.DataFrame, np.ndarray, List[List[Any]]]
"""
Represents a data structure that can behave like a DataFrame.

Includes:
- Pandas DataFrame
- NumPy ndarray
- List of lists with arbitrary elements
"""

SeriesOrDFLike = Union[pd.Series, DFLike]
"""
Represents a structure that can behave like a Pandas Series or DataFrame.

Includes:
- Pandas Series
- All types represented by `DFLike`
"""

ListOrSeriesOrDF = Union[List[Any], List[List[Any]], SeriesOrDFLike]
"""
Represents a structure that can be a list, a Pandas Series, or a DataFrame-like object.

Includes:
- List of arbitrary elements
- List of lists with arbitrary elements
- All types represented by `SeriesOrDFLike`
"""
