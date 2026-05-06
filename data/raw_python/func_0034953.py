def check_array_or_list(input):
    """Return 1D ndarray, if input can be converted and elements are
    non-negative."""
    if type(input) != np.ndarray:
        if type(input) == list:
            output = np.array(input)
        else:
            raise TypeError('Expecting input type as ndarray or list.')
    else:
        output = input

    if output.ndim != 1:
        raise ValueError('Input array must have 1 dimension.')

    if np.sum(output < 0.) > 0:
            raise ValueError("Input array values cannot be negative.")

    return output