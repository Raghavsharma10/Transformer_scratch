def consecutive(data, stepsize=1):
    """Converts array into chunks with consecutive elements of given step size.
    http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)