def calc_remotedemand_v1(self):
    """Estimate the discharge demand of a cross section far downstream.

    Required control parameter:
      |RemoteDischargeMinimum|

    Required derived parameters:
      |dam_derived.TOY|

    Required flux sequence:
      |dam_derived.TOY|

    Calculated flux sequence:
      |RemoteDemand|

    Basic equation:
      :math:`RemoteDemand =
      max(RemoteDischargeMinimum - NaturalRemoteDischarge, 0`

    Examples:

        Low water elevation is often restricted to specific month of the year.
        Sometimes the pursued lowest discharge value varies over the year
        to allow for a low flow variability that is in some agreement with
        the natural flow regime.  The HydPy-Dam model supports such
        variations.  Hence we define a short simulation time period first.
        This enables us to show how the related parameters values can be
        defined and how the calculation of the `remote` water demand
        throughout the year actually works:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'

        Prepare the dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()

        Assume the required discharge at a gauge downstream being 2 m³/s
        in the hydrological summer half-year (April to October).  In the
        winter month (November to May), there is no such requirement:

        >>> remotedischargeminimum(_11_1_12=0.0, _03_31_12=0.0,
        ...                        _04_1_12=2.0, _10_31_12=2.0)
        >>> derived.toy.update()

        Prepare a test function, that calculates the remote discharge demand
        based on the parameter values defined above and for natural remote
        discharge values ranging between 0 and 3 m³/s:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_remotedemand_v1, last_example=4,
        ...                 parseqs=(fluxes.naturalremotedischarge,
        ...                          fluxes.remotedemand))
        >>> test.nexts.naturalremotedischarge = range(4)

        On April 1, the required discharge is 2 m³/s:

        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> test()
        | ex. | naturalremotedischarge | remotedemand |
        -----------------------------------------------
        |   1 |                    0.0 |          2.0 |
        |   2 |                    1.0 |          1.0 |
        |   3 |                    2.0 |          0.0 |
        |   4 |                    3.0 |          0.0 |

        On May 31, the required discharge is 0 m³/s:

        >>> model.idx_sim = pub.timegrids.init['2001.03.31']
        >>> test()
        | ex. | naturalremotedischarge | remotedemand |
        -----------------------------------------------
        |   1 |                    0.0 |          0.0 |
        |   2 |                    1.0 |          0.0 |
        |   3 |                    2.0 |          0.0 |
        |   4 |                    3.0 |          0.0 |
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.remotedemand = max(con.remotedischargeminimum[der.toy[self.idx_sim]] -
                           flu.naturalremotedischarge, 0.)