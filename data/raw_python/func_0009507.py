def subCell2DFnArray(arr, fn, shape, dtype=None, **kwargs):
    '''
    Return array where every cell is the output of a given cell function

    Args:
       fn (function): ...to be executed on all sub-arrays

    Returns:
        array: value of every cell equals result of fn(sub-array)

    Example:    
        mx = subCell2DFnArray(myArray, np.max, (10,6) )
        - -> here mx is a 2d array containing all cell maxima
    '''

    sh = list(arr.shape)
    sh[:2] = shape
    out = np.empty(sh, dtype=dtype)
    for i, j, c in subCell2DGenerator(arr, shape, **kwargs):
        out[i, j] = fn(c)
    return out