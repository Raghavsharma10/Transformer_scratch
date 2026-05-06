def _evaluate(x, y, weights):
    '''
    get the parameters of the, needed by 'function'
    through curve fitting
    '''
    i = _validI(x, y, weights)
    xx = x[i]
    y = y[i]

    try:
        fitParams = _fit(xx, y)
        # bound noise fn to min defined y value:
        minY = function(xx[0], *fitParams)
        fitParams = np.insert(fitParams, 0, minY)
        fn = lambda x, minY=minY: boundedFunction(x, *fitParams)
    except RuntimeError:
        print(
            "couldn't fit noise function with filtered indices, use polynomial fit instead")
        fitParams = None
        fn = smooth(xx, y, weights[i])
    return fitParams, fn, i