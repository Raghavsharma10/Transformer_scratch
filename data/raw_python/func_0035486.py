def classify_harmonic(self, partial_labels, use_CMN=True):
    '''Harmonic function method for semi-supervised classification,
    also known as the Gaussian Mean Fields algorithm.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    use_CMN : when True, apply Class Mass Normalization

    From "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions"
      by Zhu, Ghahramani, and Lafferty in 2003.

    Based on the matlab code at:
      http://pages.cs.wisc.edu/~jerryzhu/pub/harmonic_function.m
    '''
    # prepare labels
    labels = np.array(partial_labels, copy=True)
    unlabeled = labels == -1

    # convert known labels to one-hot encoding
    fl, classes = _onehot(labels[~unlabeled])

    L = self.laplacian(normed=False)
    if ss.issparse(L):
      L = L.tocsr()[unlabeled].toarray()
    else:
      L = L[unlabeled]

    Lul = L[:,~unlabeled]
    Luu = L[:,unlabeled]
    fu = -np.linalg.solve(Luu, Lul.dot(fl))

    if use_CMN:
      scale = (1 + fl.sum(axis=0)) / fu.sum(axis=0)
      fu *= scale

    # assign new labels
    labels[unlabeled] = classes[fu.argmax(axis=1)]
    return labels