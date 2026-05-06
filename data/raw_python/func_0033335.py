def extract(filename, doy, latitude, longitude, depth):
    """
        For now only the nearest value
        For now only for one position, not an array of positions
        longitude 0-360
    """
    assert np.size(doy) == 1
    assert np.size(latitude) == 1
    assert np.size(longitude) == 1
    assert np.size(depth) == 1

    assert (longitude >= 0) & (longitude <= 360)
    assert depth >= 0

    nc = netCDF4.Dataset(filename)

    t = 2 * np.pi * doy/366

    Z = np.absolute(nc['depth'][:] - depth).argmin()
    I = np.absolute(nc['lat'][:] - latitude).argmin()
    J = np.absolute(nc['lon'][:] - longitude).argmin()

    # Naive solution
    value = nc['mean'][:, I, J]
    value[:64] += nc['an_cos'][Z, I, J] * np.cos(t) + \
            nc['an_sin'][:, I, J] * np.sin(t)
    value[:55] += nc['sa_cos'][Z, I, J] * np.cos(2*t) + \
            nc['sa_sin'][:, I, J] * np.sin(2*t)
    value = value[Z]

    std = nc['std_dev'][Z, I, J]

    return value, std