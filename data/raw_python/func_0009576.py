def positionToIntensityUncertaintyForPxGroup(image, std, y0, y1, x0, x1):
    '''
    like positionToIntensityUncertainty
    but calculated average uncertainty for an area [y0:y1,x0:x1]
    '''
    fy, fx = y1 - y0, x1 - x0
    if fy != fx:
        raise Exception('averaged area need to be square ATM')
    image = _coarsenImage(image, fx)
    k = _kSizeFromStd(std)
    y0 = int(round(y0 / fy))
    x0 = int(round(x0 / fx))
    arr = image[y0 - k:y0 + k, x0 - k:x0 + k]
    U = positionToIntensityUncertainty(arr, std / fx, std / fx)
    return U[k:-k, k:-k]