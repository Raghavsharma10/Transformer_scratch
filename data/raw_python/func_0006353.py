def get_not_unique_values(array):
    '''Returns the values that appear at least twice in array.

    Parameters
    ----------
    array : array like

    Returns
    -------
    numpy.array
    '''
    s = np.sort(array, axis=None)
    s = s[s[1:] == s[:-1]]
    return np.unique(s)