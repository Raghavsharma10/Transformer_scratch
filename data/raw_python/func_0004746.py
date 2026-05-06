def resample(grid, wl, flux):
    """ Resample spectrum onto desired grid """
    flux_rs = (interpolate.interp1d(wl, flux))(grid)
    return flux_rs