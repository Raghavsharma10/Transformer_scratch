def monochromaticWavelength(img):
    '''
    TODO##########
    '''
    # peak wave lengths: https://en.wikipedia.org/wiki/RGB_color_model
    out = _calc(img)

    peakWavelengths = (570, 540, 440)  # (r,g,b)
#     s = sum(peakWavelengths)
    for n, p in enumerate(peakWavelengths):
        out[..., n] *= p
    return out.sum(axis=2)