def temporalSignalStability(imgs, times, down_scale_factor=1):
    '''
    (Electroluminescence) signal is not stable over time
        especially next to cracks.
    This function takes a set of images
    and returns parameters, needed to transform uncertainty 
        to other exposure times using [adjustUncertToExposureTime]
    
    
    return [signal uncertainty] obtained from linear fit to [imgs]
           [average event length] 
           [ascent],[offset] of linear fit
    
    --------
    [imgs] --> corrected EL images captured in sequence
    
    [times] --> absolute measurement times of all [imgs]
                e.g. every image was taken every 60 sec, then 
                times=60,120,180... 
    [down_scale_factor] --> down scale [imgs] to speed up process
    -------

    More information can be found at ...
    ----
    K.Bedrich: Quantitative Electroluminescence Imaging, PhD Thesis, 2017
    Subsection 5.1.4.3: Exposure Time Dependency
    ----
    '''
    imgs = np.asarray(imgs)
    s0, s1, s2 = imgs.shape

    #down scale imgs to speed up process:
    if down_scale_factor > 1:
        s1 //= down_scale_factor
        s2 //= down_scale_factor
        imgs2 = np.empty(shape=(s0, s1, s2))
        for n, c in enumerate(imgs):
            imgs2[n] = cv2.resize(c, (s2, s1), interpolation=cv2.INTER_AREA)
        imgs = imgs2
    
    # linear fit for every point in image set:
    ascent, offset, error = linRegressUsingMasked2dArrays(
                                times, imgs, calcError=True)
    
    # functionally obtained [imgs]:
    fn_imgs = np.array([offset + t * ascent for t in times])
    #difference between [imgs] for fit result:
    diff = imgs - fn_imgs
    diff = median_filter(diff, 5)

    error_t = np.tile(error, (s0, 1, 1))
    # find events: 
    evt = (np.abs(diff) > 0.5 * error_t) 
    # calc average event length:
    avlen = _calcAvgLen(evt, np.empty(shape=evt.shape[1:]))
    
    #cannot calc event length smaller exposure time, so:
    i = avlen == 0
    avlen = maskedFilter(avlen, mask=i, fn='mean', ksize=7, fill_mask=False)
    # remove single px:
    i = maximum_filter(i, 3)
    avlen[i] = 0
    avlen = maximum_filter(avlen, 3)

    i = avlen == 0
    avlen = median_filter(avlen, 3)
    avlen[i] = 0

    return error, avlen, ascent, offset