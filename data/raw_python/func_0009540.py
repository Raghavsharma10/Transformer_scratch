def estimateBackgroundLevel(img, image_is_artefact_free=False, 
                            min_rel_size=0.05, max_abs_size=11):
    '''
    estimate background level through finding the most homogeneous area
    and take its average
    
    min_size - relative size of the examined area
    '''

    s0,s1 = img.shape[:2]
    s = min(max_abs_size, int(max(s0,s1)*min_rel_size))
    arr = np.zeros(shape=(s0-2*s, s1-2*s), dtype=img.dtype)
    
    #fill arr:
    _spatialStd(img, arr, s)
    #most homogeneous area:
    i,j = np.unravel_index(arr.argmin(), arr.shape)
    sub = img[int(i+0.5*s):int(i+s*1.5), 
              int(j+s*0.5):int(j+s*1.5)]

    return np.median(sub)