def maskedFilter(arr, mask, ksize=30, fill_mask=True,
                 fn='median'):
    '''
    fn['mean', 'median']

    fill_mask=True:
        replaced masked areas with filtered results

    fill_mask=False:
    masked areas are ignored
    '''

    if fill_mask:
        mask1 = mask
        out = arr
    else:
        mask1 = ~mask
        out = np.full_like(arr, fill_value=np.nan)
    mask2 = ~mask

    if fn == 'mean':
        _calcMean(arr, mask1, mask2, out, ksize // 2)
    else:
        buff = np.empty(shape=(ksize * ksize), dtype=arr.dtype)
        _calcMedian(arr, mask1, mask2, out, ksize // 2, buff)
    return out