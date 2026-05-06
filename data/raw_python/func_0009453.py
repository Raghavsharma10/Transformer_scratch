def rgChromaticity(img):
    '''
    returns the normalized RGB space (RGB/intensity)
    see https://en.wikipedia.org/wiki/Rg_chromaticity
    '''
    out = _calc(img)
    if img.dtype == np.uint8:
        out = (255 * out).astype(np.uint8)
    return out