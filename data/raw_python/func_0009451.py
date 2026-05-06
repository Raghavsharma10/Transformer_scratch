def toNoUintArray(arr):
    '''
    cast array to the next higher integer array
    if dtype=unsigned integer
    '''
    d = arr.dtype
    if d.kind == 'u':
        arr = arr.astype({1: np.int16,
                          2: np.int32,
                          4: np.int64}[d.itemsize])
    return arr