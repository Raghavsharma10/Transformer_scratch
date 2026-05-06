def update_loggedtotalremotedischarge_v1(self):
    """Log a new entry of discharge at a cross section far downstream.

    Required control parameter:
      |NmbLogEntries|

    Required flux sequence:
      |TotalRemoteDischarge|

    Calculated flux sequence:
      |LoggedTotalRemoteDischarge|

    Example:

        The following example shows that, with each new method call, the
        three memorized values are successively moved to the right and the
        respective new value is stored on the bare left position:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> nmblogentries(3)
        >>> logs.loggedtotalremotedischarge = 0.0
        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.update_loggedtotalremotedischarge_v1,
        ...                 last_example=4,
        ...                 parseqs=(fluxes.totalremotedischarge,
        ...                          logs.loggedtotalremotedischarge))
        >>> test.nexts.totalremotedischarge = [1., 3., 2., 4]
        >>> del test.inits.loggedtotalremotedischarge
        >>> test()
        | ex. | totalremotedischarge |           loggedtotalremotedischarge |
        ---------------------------------------------------------------------
        |   1 |                  1.0 | 1.0  0.0                         0.0 |
        |   2 |                  3.0 | 3.0  1.0                         0.0 |
        |   3 |                  2.0 | 2.0  3.0                         1.0 |
        |   4 |                  4.0 | 4.0  2.0                         3.0 |
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(con.nmblogentries-1, 0, -1):
        log.loggedtotalremotedischarge[idx] = \
                                log.loggedtotalremotedischarge[idx-1]
    log.loggedtotalremotedischarge[0] = flu.totalremotedischarge