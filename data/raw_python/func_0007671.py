def equal_distribution_folds(y, folds=2):
    """Creates `folds` number of indices that has roughly balanced multi-label distribution.

    Args:
        y: The multi-label outputs.
        folds: The number of folds to create.

    Returns:
        `folds` number of indices that have roughly equal multi-label distributions.
    """
    n, classes = y.shape

    # Compute sample distribution over classes
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()

    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(n):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(
                np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)

    logger.debug("Fold distributions:")
    logger.debug(fold_dist)
    return index_list