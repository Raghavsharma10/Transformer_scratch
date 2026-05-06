def calc_login_v1(self):
    """Refresh the input log sequence for the different MA processes.

    Required derived parameters:
      |Nmb|
      |MA_Order|

    Required flux sequence:
      |QPIn|

    Updated log sequence:
      |LogIn|

    Example:

        Assume there are three response functions, involving one, two and
        three MA coefficients respectively:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(3)
        >>> derived.ma_order.shape = 3
        >>> derived.ma_order = 1, 2, 3
        >>> fluxes.qpin.shape = 3
        >>> logs.login.shape = (3, 3)

        The "memory values" of the different MA processes are defined as
        follows (one row for each process):

        >>> logs.login = ((1.0, nan, nan),
        ...               (2.0, 3.0, nan),
        ...               (4.0, 5.0, 6.0))

        These are the new inflow discharge portions to be included into
        the memories of the different processes:

        >>> fluxes.qpin = 7.0, 8.0, 9.0

        Through applying method |calc_login_v1| all values already
        existing are shifted to the right ("into the past").  Values,
        which are no longer required due to the limited order or the
        different MA processes, are discarded.  The new values are
        inserted in the first column:

        >>> model.calc_login_v1()
        >>> logs.login
        login([[7.0, nan, nan],
               [8.0, 2.0, nan],
               [9.0, 4.0, 5.0]])
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(der.nmb):
        for jdx in range(der.ma_order[idx]-2, -1, -1):
            log.login[idx, jdx+1] = log.login[idx, jdx]
    for idx in range(der.nmb):
        log.login[idx, 0] = flu.qpin[idx]