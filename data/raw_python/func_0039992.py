def catvl(z, ver, vnew, lamb, lambnew, br):
    """
    trapz integrates over altitude axis, axis = -2
    concatenate over reaction dimension, axis = -1

    br: column integrated brightness
    lamb: wavelength [nm]
    ver: volume emission rate  [photons / cm^-3 s^-3 ...]
    """
    if ver is not None:
        br = np.concatenate((br, np.trapz(vnew, z, axis=-2)), axis=-1)  # must come first!
        ver = np.concatenate((ver, vnew), axis=-1)
        lamb = np.concatenate((lamb, lambnew))
    else:
        ver = vnew.copy(order='F')
        lamb = lambnew.copy()
        br = np.trapz(ver, z, axis=-2)

    return ver, lamb, br