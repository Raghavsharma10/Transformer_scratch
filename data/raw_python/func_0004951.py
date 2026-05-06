def angledependentabsorption(twotheta, transmission):
    """Correction for angle-dependent absorption of the sample

    Inputs:
        twotheta: matrix of two-theta values
        transmission: the transmission of the sample (I_after/I_before, or
            exp(-mu*d))

    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it. Note, that this does not corrects for
        sample transmission by itself, as the 2*theta -> 0 limit of this matrix
        is unity. Twotheta==0 and transmission==1 cases are handled correctly
        (the limit is 1 in both cases).
    """
    cor = np.ones(twotheta.shape)
    if transmission == 1:
        return cor
    mud = -np.log(transmission)

    cor[twotheta > 0] = transmission * mud * (1 - 1 / np.cos(twotheta[twotheta > 0])) / (np.exp(-mud / np.cos(twotheta[twotheta > 0])) - np.exp(-mud))
    return cor