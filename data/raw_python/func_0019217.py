def update_loggedoutflow_v1(self):
    """Log a new entry of discharge at a cross section far downstream.

    Required control parameter:
      |NmbLogEntries|

    Required flux sequence:
      |Outflow|

    Calculated flux sequence:
      |LoggedOutflow|

    Example:

        The following example shows that, with each new method call, the
        three memorized values are successively moved to the right and the
        respective new value is stored on the bare left position:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> nmblogentries(3)
        >>> logs.loggedoutflow = 0.0
        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.update_loggedoutflow_v1,
        ...                 last_example=4,
        ...                 parseqs=(fluxes.outflow,
        ...                          logs.loggedoutflow))
        >>> test.nexts.outflow = [1.0, 3.0, 2.0, 4.0]
        >>> del test.inits.loggedoutflow
        >>> test()
        | ex. | outflow |           loggedoutflow |
        -------------------------------------------
        |   1 |     1.0 | 1.0  0.0            0.0 |
        |   2 |     3.0 | 3.0  1.0            0.0 |
        |   3 |     2.0 | 2.0  3.0            1.0 |
        |   4 |     4.0 | 4.0  2.0            3.0 |
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for idx in range(con.nmblogentries-1, 0, -1):
        log.loggedoutflow[idx] = log.loggedoutflow[idx-1]
    log.loggedoutflow[0] = flu.outflow