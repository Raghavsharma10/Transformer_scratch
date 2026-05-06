def fastFilter(arr, ksize=30, every=None, resize=True, fn='median',
                  interpolation=cv2.INTER_LANCZOS4,
                  smoothksize=0,
                  borderMode=cv2.BORDER_REFLECT):
    '''
    fn['nanmean', 'mean', 'nanmedian', 'median']
    
    a fast 2d filter for large kernel sizes that also 
    works with nans
    the computation speed is increased because only 'every'nsth position 
    within the median kernel is evaluated
    '''
    if every is None:
        every = max(ksize//3, 1)
    else:
        assert ksize >= 3*every
    s0,s1 = arr.shape[:2]
    
    ss0 = s0//every
    every = s0//ss0
    ss1 = s1//every
    
    out = np.full((ss0+1,ss1+1), np.nan)
    
    c = {'median':_calcMedian,
         'nanmedian':_calcNanMedian,
         'nanmean':_calcNanMean,
         'mean':_calcMean,
         }[fn]
    ss0,ss1 = c(arr, out, ksize, every)
    out = out[:ss0,:ss1]
    
    if smoothksize:
        out = gaussian_filter(out, smoothksize)
        
    
    if not resize:
        return out
    return cv2.resize(out, arr.shape[:2][::-1],
               interpolation=interpolation)