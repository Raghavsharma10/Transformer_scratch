def flatFieldFromCloseDistance(imgs, bg_imgs=None):
    '''
    Average multiple images of a homogeneous device
    imaged directly in front the camera lens.

    if [bg_imgs] are not given, background level is extracted
        from 1% of the cumulative intensity distribution
        of the averaged [imgs]

    This measurement method is referred as 'Method A' in
    ---
    K.Bedrich, M.Bokalic et al.:
    ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
    ADVANCED FLAT FIELD CALIBRATION,2017
    ---
    '''
    img = imgAverage(imgs)
    bg = getBackground2(bg_imgs, img)
    img -= bg
    img = toGray(img)
    mx = median_filter(img[::10, ::10], 3).max()
    img /= mx
    return img