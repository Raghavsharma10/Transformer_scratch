def calc_dist(lamost_point, training_points, coeffs):
    """ avg dist from one lamost point to nearest 10 training points """
    diff2 = (training_points - lamost_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs, axis=1))
    return np.mean(dist[dist.argsort()][0:10])