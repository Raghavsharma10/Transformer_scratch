def nan_maximum_filter(arr, ksize):
    '''
    same as scipy.filters.maximum_filter
    but working excluding nans
    '''
    out = np.empty_like(arr)
    _calc(arr, out, ksize//2)
    return out