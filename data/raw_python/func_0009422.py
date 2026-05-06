def vignettingFromDifferentObjects(imgs, bg):
    '''
    Extract vignetting from a set of images
    containing different devices
    The devices spatial inhomogeneities are averaged

    This method is referred as 'Method C' in 
    ---
    K.Bedrich, M.Bokalic et al.:
    ELECTROLUMINESCENCE IMAGING OF PV DEVICES:
    ADVANCED FLAT FIELD CALIBRATION,2017
    ---
    '''

    f = FlatFieldFromImgFit(imgs, bg)
    return f.result, f.mask