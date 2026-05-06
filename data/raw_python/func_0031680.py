def normalize(data):
    """
    Function to normalize data to have mean 0 and unity standard deviation
    (also called z-transform)
    
    
    Parameters
    ----------
    data : numpy.ndarray
    
    
    Returns
    -------
    numpy.ndarray
        z-transform of input array
    
    """
    data = data.astype(float)
    data -= data.mean()
    
    return data / data.std()