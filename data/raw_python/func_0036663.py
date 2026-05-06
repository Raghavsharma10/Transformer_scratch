def L_fc(fdata):
    """Apply L in the Fourier domain."""

    fd = np.copy(fdata)

    dphi_fc(fdata)
    divsin_fc(fdata)

    dtheta_fc(fd)

    return (1j * fdata, -1j * fd)