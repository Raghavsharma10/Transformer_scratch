def _cdist_scipy(x, y, exponent=1):
    """Pairwise distance between the points in two sets."""
    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = _spatial.distance.cdist(x, y, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances