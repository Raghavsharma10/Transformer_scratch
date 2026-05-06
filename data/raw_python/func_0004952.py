def angledependentabsorption_errorprop(twotheta, dtwotheta, transmission, dtransmission):
    """Correction for angle-dependent absorption of the sample with error propagation

    Inputs:
        twotheta: matrix of two-theta values
        dtwotheta: matrix of absolute error of two-theta values
        transmission: the transmission of the sample (I_after/I_before, or
            exp(-mu*d))
        dtransmission: the absolute error of the transmission of the sample

    Two matrices are returned: the first one is the correction (intensity matrix
        should be multiplied by it), the second is its absolute error.
    """
    # error propagation formula calculated using sympy
    return (angledependentabsorption(twotheta, transmission),
            _calc_angledependentabsorption_error(twotheta, dtwotheta, transmission, dtransmission))