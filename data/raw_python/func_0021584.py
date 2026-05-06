def get_delta(D, k):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in range(k):
        result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    return result