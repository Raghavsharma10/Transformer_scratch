def _findNearest(arr, value):
    """ Finds the value in arr that value is closest to
    """
    arr = np.array(arr)
    # find nearest value in array
    idx = (abs(arr-value)).argmin()
    return arr[idx]