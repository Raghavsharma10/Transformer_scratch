def r(w, Xs):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    c = C(w, Xs)

    return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)