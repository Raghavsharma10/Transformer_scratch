def oneImageNLF(img, img2=None, signal=None):
    '''
    Estimate the NLF from one or two images of the same kind
    '''
    x, y, weights, signal = calcNLF(img, img2, signal)
    _, fn, _ = _evaluate(x, y, weights)
    return fn, signal