def calc_naturalremotedischarge_v1(self):
    """Try to estimate the natural discharge of a cross section far downstream
    based on the last few simulation steps.

    Required control parameter:
      |NmbLogEntries|

    Required log sequences:
      |LoggedTotalRemoteDischarge|
      |LoggedOutflow|

    Calculated flux sequence:
      |NaturalRemoteDischarge|

    Basic equation:
      :math:`RemoteDemand =
      max(\\frac{\\Sigma(LoggedTotalRemoteDischarge - LoggedOutflow)}
      {NmbLogEntries}), 0)`

    Examples:

        Usually, the mean total remote flow should be larger than the mean
        dam outflows.  Then the estimated natural remote discharge is simply
        the difference of both mean values:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> nmblogentries(3)
        >>> logs.loggedtotalremotedischarge(2.5, 2.0, 1.5)
        >>> logs.loggedoutflow(2.0, 1.0, 0.0)
        >>> model.calc_naturalremotedischarge_v1()
        >>> fluxes.naturalremotedischarge
        naturalremotedischarge(1.0)

        Due to the wave travel times, the difference between remote discharge
        and dam outflow mights sometimes be negative.  To avoid negative
        estimates of natural discharge, it its value is set to zero in
        such cases:

        >>> logs.loggedoutflow(4.0, 3.0, 5.0)
        >>> model.calc_naturalremotedischarge_v1()
        >>> fluxes.naturalremotedischarge
        naturalremotedischarge(0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    flu.naturalremotedischarge = 0.
    for idx in range(con.nmblogentries):
        flu.naturalremotedischarge += (
            log.loggedtotalremotedischarge[idx] - log.loggedoutflow[idx])
    if flu.naturalremotedischarge > 0.:
        flu.naturalremotedischarge /= con.nmblogentries
    else:
        flu.naturalremotedischarge = 0.