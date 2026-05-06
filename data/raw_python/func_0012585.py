def randsample(vec, nr_samples, with_replacement = False):
    """
    Draws nr_samples random samples from vec.
    """
    if not with_replacement:
        return np.random.permutation(vec)[0:nr_samples]
    else:
        return np.asarray(vec)[np.random.randint(0, len(vec), nr_samples)]