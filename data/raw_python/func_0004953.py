def angledependentairtransmission(twotheta, mu_air, sampletodetectordistance):
    """Correction for the angle dependent absorption of air in the scattered
    beam path.

    Inputs:
            twotheta: matrix of two-theta values
            mu_air: the linear absorption coefficient of air
            sampletodetectordistance: sample-to-detector distance

    1/mu_air and sampletodetectordistance should have the same dimension

    The scattering intensity matrix should be multiplied by the resulting
    correction matrix."""
    return np.exp(mu_air * sampletodetectordistance / np.cos(twotheta))