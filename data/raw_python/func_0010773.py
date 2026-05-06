def distribute_equally(daily_data, divide=False):
    """Obtains hourly values by equally distributing the daily values.

    Args:
        daily_data: daily values
        divide: if True, divide resulting values by the number of hours in
            order to preserve the daily sum (required e.g. for precipitation).

    Returns:
        Equally distributed hourly values.
    """

    index = hourly_index(daily_data.index)
    hourly_data = daily_data.reindex(index)
    hourly_data = hourly_data.groupby(hourly_data.index.day).transform(
        lambda x: x.fillna(method='ffill', limit=23))

    if divide:
        hourly_data /= 24

    return hourly_data