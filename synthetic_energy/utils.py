import pandas as pd


def has_datetime_columns(dataframe: pd.DataFrame) -> bool:
    return any(
        dataframe[column].dtype
        in [
            "datetime64",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "timedelta64",
            "timedelta64[ns]",
        ]
        for column in dataframe.columns
    )


def is_time_series(dataframe: pd.DataFrame) -> bool:
    # Checks if there is a datetime column.
    datetime_columns = dataframe.select_dtypes(
        include=[pd.DatetimeTZDtype, "datetime64[ns]"]
    ).columns
    if len(datetime_columns) == 0:
        return False

    # Assumes the first datetime column is the one to check.
    datetime_column = datetime_columns[0]

    # Checks if the dates are in sequential order.
    dates = dataframe[datetime_column]
    return dates.is_monotonic_increasing or dates.is_monotonic_decreasing
