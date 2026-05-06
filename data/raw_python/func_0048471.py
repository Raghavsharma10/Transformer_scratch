def check_array(array):
    "Converts to flattened numpy arrays and ensures its not empty."

    if len(array) < 1:
        raise ValueError('Input array is empty! Must have atleast 1 element.')

    return np.ma.masked_invalid(array).flatten()