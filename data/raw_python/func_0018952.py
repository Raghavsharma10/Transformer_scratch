def calc_qma_v1(self):
    """Calculate the discharge responses of the different MA processes.

    Required derived parameters:
      |Nmb|
      |MA_Order|
      |MA_Coefs|

    Required log sequence:
      |LogIn|

    Calculated flux sequence:
      |QMA|

    Examples:

        Assume there are three response functions, involving one, two and
        three MA coefficients respectively:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(3)
        >>> derived.ma_order.shape = 3
        >>> derived.ma_order = 1, 2, 3
        >>> derived.ma_coefs.shape = (3, 3)
        >>> logs.login.shape = (3, 3)
        >>> fluxes.qma.shape = 3

        The coefficients of the different MA processes are stored in
        separate rows of the 2-dimensional parameter `ma_coefs`:

        >>> derived.ma_coefs = ((1.0, nan, nan),
        ...                     (0.8, 0.2, nan),
        ...                     (0.5, 0.3, 0.2))

        The "memory values" of the different MA processes are defined as
        follows (one row for each process).  The current values are stored
        in first column, the values of the last time step in the second
        column, and so on:

        >>> logs.login = ((1.0, nan, nan),
        ...               (2.0, 3.0, nan),
        ...               (4.0, 5.0, 6.0))

        Applying method |calc_qma_v1| is equivalent to calculating the
        inner product of the different rows of both matrices:

        >>> model.calc_qma_v1()
        >>> fluxes.qma
        qma(1.0, 2.2, 4.7)

    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(der.nmb):
        flu.qma[idx] = 0.
        for jdx in range(der.ma_order[idx]):
            flu.qma[idx] += der.ma_coefs[idx, jdx] * log.login[idx, jdx]