def angledependentairtransmission_errorprop(twotheta, dtwotheta, mu_air,
                                            dmu_air, sampletodetectordistance,
                                            dsampletodetectordistance):
    """Correction for the angle dependent absorption of air in the scattered
    beam path, with error propagation

    Inputs:
            twotheta: matrix of two-theta values
            dtwotheta: absolute error matrix of two-theta
            mu_air: the linear absorption coefficient of air
            dmu_air: error of the linear absorption coefficient of air
            sampletodetectordistance: sample-to-detector distance
            dsampletodetectordistance: error of the sample-to-detector distance

    1/mu_air and sampletodetectordistance should have the same dimension

    The scattering intensity matrix should be multiplied by the resulting
    correction matrix."""
    return (np.exp(mu_air * sampletodetectordistance / np.cos(twotheta)),
            np.sqrt(dmu_air ** 2 * sampletodetectordistance ** 2 *
                    np.exp(2 * mu_air * sampletodetectordistance / np.cos(twotheta))
                    / np.cos(twotheta) ** 2 + dsampletodetectordistance ** 2 *
                    mu_air ** 2 * np.exp(2 * mu_air * sampletodetectordistance /
                                         np.cos(twotheta)) /
                    np.cos(twotheta) ** 2 + dtwotheta ** 2 * mu_air ** 2 *
                    sampletodetectordistance ** 2 *
                    np.exp(2 * mu_air * sampletodetectordistance / np.cos(twotheta))
                     * np.sin(twotheta) ** 2 / np.cos(twotheta) ** 4)
            )