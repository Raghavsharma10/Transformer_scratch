def progression_sinusoidal(week, start_weight, final_weight, start_week,
                           end_week,
                           periods=2, scale=0.025, offset=0):
    """A sinusoidal progression function going through the points
    ('start_week', 'start_weight') and ('end_week', 'final_weight'), evaluated
    in 'week'. This function calls a linear progression function
    and multiplies it by a sinusoid.

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
    periods
        Number of sinusoidal periods in the time range.
    scale
        The scale (amplitude) of the sinusoidal term.
    offset
        The offset (shift) of the sinusoid.


    Returns
    -------
    weight
        The weight at 'week'.


    Examples
    -------
    >>> progression_sinusoidal(1, 100, 120, 1, 8)
    100.0
    >>> progression_sinusoidal(8, 100, 120, 1, 8)
    120.0
    >>> progression_sinusoidal(4, 100, 120, 1, 8)
    106.44931454758678
    """
    # Get the linear model
    linear = progression_linear(week, start_weight, final_weight,
                                start_week, end_week)

    # Calculate the time period and the argument to the sine function
    time_period = end_week - start_week
    sine_argument = ((week - offset - start_week) * (math.pi * 2) /
                     (time_period / periods))

    linear_with_sinusoidal = linear * (1 + scale * math.sin(sine_argument))
    return linear_with_sinusoidal