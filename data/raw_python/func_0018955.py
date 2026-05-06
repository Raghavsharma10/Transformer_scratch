def calc_logout_v1(self):
    """Refresh the log sequence for the different AR processes.

    Required derived parameters:
      |Nmb|
      |AR_Order|

    Required flux sequence:
      |QPOut|

    Updated log sequence:
      |LogOut|

    Example:

        Assume there are four response functions, involving zero, one, two
        and three AR coefficients respectively:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(4)
        >>> derived.ar_order.shape = 4
        >>> derived.ar_order = 0, 1, 2, 3
        >>> fluxes.qpout.shape = 4
        >>> logs.logout.shape = (4, 3)

        The "memory values" of the different AR processes are defined as
        follows (one row for each process).  Note the special case of the
        first AR process of zero order (first row), which is why there are
        no autoregressive memory values required:

        >>> logs.logout = ((nan, nan, nan),
        ...                (0.0, nan, nan),
        ...                (1.0, 2.0, nan),
        ...                (3.0, 4.0, 5.0))

        These are the new outflow discharge portions to be included into
        the memories of the different processes:

        >>> fluxes.qpout = 6.0, 7.0, 8.0, 9.0

        Through applying method |calc_logout_v1| all values already
        existing are shifted to the right ("into the past").  Values, which
        are no longer required due to the limited order or the different
        AR processes, are discarded.  The new values are inserted in the
        first column:

        >>> model.calc_logout_v1()
        >>> logs.logout
        logout([[nan, nan, nan],
                [7.0, nan, nan],
                [8.0, 1.0, nan],
                [9.0, 3.0, 4.0]])

    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(der.nmb):
        for jdx in range(der.ar_order[idx]-2, -1, -1):
            log.logout[idx, jdx+1] = log.logout[idx, jdx]
    for idx in range(der.nmb):
        if der.ar_order[idx] > 0:
            log.logout[idx, 0] = flu.qpout[idx]