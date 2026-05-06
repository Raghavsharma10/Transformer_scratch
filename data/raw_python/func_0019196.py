def calc_remotefailure_v1(self):
    """Estimate the shortfall of actual discharge under the required discharge
    of a cross section far downstream.

    Required control parameters:
      |NmbLogEntries|
      |RemoteDischargeMinimum|

    Required derived parameters:
      |dam_derived.TOY|

    Required log sequence:
      |LoggedTotalRemoteDischarge|

    Calculated flux sequence:
      |RemoteFailure|

    Basic equation:
      :math:`RemoteFailure =
      \\frac{\\Sigma(LoggedTotalRemoteDischarge)}{NmbLogEntries} -
      RemoteDischargeMinimum`

    Examples:

        As explained in the documentation on method |calc_remotedemand_v1|,
        we have to define a simulation period first:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'

        Now we prepare a dam model with log sequences memorizing three values:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> nmblogentries(3)

        Again, the required discharge is 2 m³/s in summer and 0 m³/s in winter:

        >>> remotedischargeminimum(_11_1_12=0.0, _03_31_12=0.0,
        ...                        _04_1_12=2.0, _10_31_12=2.0)
        >>> derived.toy.update()

        Let it be supposed that the actual discharge at the remote
        cross section droped from 2 m³/s to 0  m³/s over the last three days:

        >>> logs.loggedtotalremotedischarge(0.0, 1.0, 2.0)

        This means that for the April 1 there would have been an averaged
        shortfall of 1 m³/s:

        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> model.calc_remotefailure_v1()
        >>> fluxes.remotefailure
        remotefailure(1.0)

        Instead for May 31 there would have been an excess of 1 m³/s, which
        is interpreted to be a "negative failure":

        >>> model.idx_sim = pub.timegrids.init['2001.03.31']
        >>> model.calc_remotefailure_v1()
        >>> fluxes.remotefailure
        remotefailure(-1.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    flu.remotefailure = 0
    for idx in range(con.nmblogentries):
        flu.remotefailure -= log.loggedtotalremotedischarge[idx]
    flu.remotefailure /= con.nmblogentries
    flu.remotefailure += con.remotedischargeminimum[der.toy[self.idx_sim]]