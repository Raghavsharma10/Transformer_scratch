def _cosine_function(x, a, b, t_shift):
    """genrates a diurnal course of windspeed accroding to the cosine function

    Args:
        x: series of euqally distributed windspeed values
        a: parameter a for the cosine function
        b: parameter b for the cosine function
        t_shift: parameter t_shift for the cosine function
        
    Returns:
        series including diurnal course of windspeed.
    """

    mean_wind, t = x
    return a * mean_wind * np.cos(np.pi * (t - t_shift) / 12) + b * mean_wind