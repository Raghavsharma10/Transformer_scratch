def weighted_std(values, weights):
    """ Calculate standard deviation weighted by errors """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)