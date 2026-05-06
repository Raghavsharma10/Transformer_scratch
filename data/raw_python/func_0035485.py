def classify_lgc(self, partial_labels, kernel='rbf', alpha=0.2, tol=1e-3,
                   max_iter=30):
    '''Iterative label spreading for semi-supervised classification.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    kernel: one of {'none', 'rbf', 'binary'}, for reweighting edges.
    alpha: scalar, clamping factor.
    tol: scalar, convergence tolerance.
    max_iter: integer, cap on the number of iterations performed.

    From "Learning with local and global consistency"
      by Zhou et al. in 2004.

    Based on the LabelSpreading implementation in scikit-learn.
    '''
    # compute the gram matrix
    gram = -self.kernelize(kernel).laplacian(normed=True)
    if ss.issparse(gram):
      gram.data[gram.row == gram.col] = 0
    else:
      np.fill_diagonal(gram, 0)

    # initialize label distributions
    partial_labels = np.asarray(partial_labels)
    unlabeled = partial_labels == -1
    label_dists, classes = _onehot(partial_labels, mask=~unlabeled)

    # initialize clamping terms
    clamp_weights = np.where(unlabeled, alpha, 1)[:,None]
    y_static = label_dists * min(1 - alpha, 1)

    # iterate
    for it in range(max_iter):
      old_label_dists = label_dists
      label_dists = gram.dot(label_dists)
      label_dists *= clamp_weights
      label_dists += y_static
      # check convergence
      if np.abs(label_dists - old_label_dists).sum() <= tol:
        break
    else:
      warnings.warn("classify_lgc didn't converge in %d iterations" % max_iter)

    return classes[label_dists.argmax(axis=1)]