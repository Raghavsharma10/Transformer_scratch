def smooth(x, y, weights):
    '''
    in case the NLF cannot be described by 
    a square root function
    commit bounded polynomial interpolation
    '''
    # Spline hard to smooth properly, therefore solfed with
    # bounded polynomal interpolation
    # ext=3: no extrapolation, but boundary value
#     return UnivariateSpline(x, y, w=weights,
#                             s=len(y)*weights.max()*100, ext=3)

#     return np.poly1d(np.polyfit(x,y,w=weights,deg=2))
    p = np.polyfit(x, y, w=weights, deg=2)
    if np.any(np.isnan(p)):
        # couldn't even do polynomial fit
        # as last option: assume constant noise
        my = np.average(y, weights=weights)
        return lambda x: my
    return lambda xint: np.poly1d(p)(np.clip(xint, x[0], x[-1]))