def _toVec(shape, val):
    '''
    takes a single value and creates a vecotor / matrix with that value filled
    in it
    '''
    mat = np.empty(shape)
    mat.fill(val)
    return mat