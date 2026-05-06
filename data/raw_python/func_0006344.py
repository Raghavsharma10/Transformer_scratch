def get_profile_histogram(x, y, n_bins=100):
    '''Takes 2D point data (x,y) and creates a profile histogram similar to the TProfile in ROOT. It calculates
    the y mean for every bin at the bin center and gives the y mean error as error bars.

    Parameters
    ----------
    x : array like
        data x positions
    y : array like
        data y positions
    n_bins : int
        the number of bins used to create the histogram
    '''
    if len(x) != len(y):
        raise ValueError('x and y dimensions have to be the same')
    y = y.astype(np.float32)
    n, bin_edges = np.histogram(x, bins=n_bins)  # needed to calculate the number of points per bin
    sy = np.histogram(x, bins=n_bins, weights=y)[0]  # the sum of the bin values
    sy2 = np.histogram(x, bins=n_bins, weights=y * y)[0]  # the quadratic sum of the bin values
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # calculate the bin center for all bins
    mean = sy / n  # calculate the mean of all bins
    std = np.sqrt((sy2 / n - mean * mean))  # TODO: not understood, need check if this is really the standard deviation
    std_mean = std / np.sqrt((n - 1))
    mean[np.isnan(mean)] = 0.
    std_mean[np.isnan(std_mean)] = 0.
    return bin_centers, mean, std_mean