def progression_linear(week, start_weight, final_weight, start_week, end_week):
    """A linear progression function going through the points
    ('start_week', 'start_weight') and ('end_week', 'final_weight'), evaluated
    in 'week'.

    Parameters
    ----------
    week
        The week to evaluate the linear function at.
    start_weight
        The weight at 'start_week'.
    final_weight
        The weight at 'end_week'.
    start_week
        The number of the first week, typically 1.
    end_week
        The number of the final week, e.g. 8.


    Returns
    -------
    weight
        The weight at 'week'.


    Examples
    -------
    >>> progression_linear(week = 2, start_weight = 100, final_weight = 120,
    ...                    start_week = 1, end_week = 3)
    110.0
    
    >>> progression_linear(3, 100, 140, 1, 5)
    120.0
    """
    # Calculate the slope of the linear function
    slope = (start_weight - final_weight) / (start_week - end_week)

    # Return the answer y = slope (x - x_0) + y_0
    return slope * (week - start_week) + start_weight