def rescale_data(data, data_min, data_max, out_min=0.0, out_max=100.0):
    """
    Rescale your input data so that is ranges over integer values, which will perform better in the watershed.

    Args:
        data: 2D or 3D ndarray being rescaled
        data_min: minimum value of input data for scaling purposes
        data_max: maximum value of input data for scaling purposes
        out_min: minimum value of scaled data
        out_max: maximum value of scaled data

    Returns:
        Linearly scaled ndarray
    """
    return (out_max - out_min) / (data_max - data_min) * (data - data_min) + out_min