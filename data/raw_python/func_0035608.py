def get_pattern_mat(oracle, pattern):
    """Output a matrix containing patterns in rows from a vmo.

    :param oracle: input vmo object
    :param pattern: pattern extracted from oracle
    :return: a numpy matrix that could be used to visualize the pattern extracted.
    """

    pattern_mat = np.zeros((len(pattern), oracle.n_states-1))
    for i,p in enumerate(pattern):
        length = p[1]
        for s in p[0]:
            pattern_mat[i][s-length:s-1] = 1
    
    return pattern_mat