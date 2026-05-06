def doBandTrapz(Aein, lambnew, fc, kin, lamb, ver, z, br):
    """
    ver dimensions: wavelength, altitude, time

     A and lambda dimensions:
    axis 0 is upper state vib. level (nu')
    axis 1 is bottom state vib level (nu'')
    there is a Franck-Condon parameter (variable fc) for each upper state nu'
    """
    tau = 1/np.nansum(Aein, axis=1)

    scalevec = (Aein * tau[:, None] * fc[:, None]).ravel(order='F')

    vnew = scalevec[None, None, :]*kin.values[..., None]

    return catvl(z, ver, vnew, lamb, lambnew, br)