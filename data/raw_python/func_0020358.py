def skew_matrix(w):
    '''Return the skew matrix of a direction w.'''
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])