def calc_qar_v1(self):
    """Calculate the discharge responses of the different AR processes.

    Required derived parameters:
      |Nmb|
      |AR_Order|
      |AR_Coefs|

    Required log sequence:
      |LogOut|

    Calculated flux sequence:
      |QAR|

    Examples:

        Assume there are four response functions, involving zero, one, two,
        and three AR coefficients respectively:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(4)
        >>> derived.ar_order.shape = 4
        >>> derived.ar_order = 0, 1, 2, 3
        >>> derived.ar_coefs.shape = (4, 3)
        >>> logs.logout.shape = (4, 3)
        >>> fluxes.qar.shape = 4

        The coefficients of the different AR processes are stored in
        separate rows of the 2-dimensional parameter `ma_coefs`.
        Note the special case of the first AR process of zero order
        (first row), which involves no autoregressive memory at all:

        >>> derived.ar_coefs = ((nan, nan, nan),
        ...                     (1.0, nan, nan),
        ...                     (0.8, 0.2, nan),
        ...                     (0.5, 0.3, 0.2))

        The "memory values" of the different AR processes are defined as
        follows (one row for each process).  The values of the last time
        step are stored in first column, the values of the last time step
        in the second column, and so on:

        >>> logs.logout = ((nan, nan, nan),
        ...                (1.0, nan, nan),
        ...                (2.0, 3.0, nan),
        ...                (4.0, 5.0, 6.0))

        Applying method |calc_qar_v1| is equivalent to calculating the
        inner product of the different rows of both matrices:

        >>> model.calc_qar_v1()
        >>> fluxes.qar
        qar(0.0, 1.0, 2.2, 4.7)

    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(der.nmb):
        flu.qar[idx] = 0.
        for jdx in range(der.ar_order[idx]):
            flu.qar[idx] += der.ar_coefs[idx, jdx] * log.logout[idx, jdx]