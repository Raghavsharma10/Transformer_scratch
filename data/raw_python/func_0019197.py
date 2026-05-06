def calc_requiredremoterelease_v1(self):
    """Guess the required release necessary to not fall below the threshold
    value at a cross section far downstream with a certain level of certainty.

    Required control parameter:
      |RemoteDischargeSafety|

    Required derived parameters:
      |RemoteDischargeSmoothPar|
      |dam_derived.TOY|

    Required flux sequence:
      |RemoteDemand|
      |RemoteFailure|

    Calculated flux sequence:
      |RequiredRemoteRelease|

    Basic equation:
      :math:`RequiredRemoteRelease = RemoteDemand + RemoteDischargeSafety
      \\cdot smooth_{logistic1}(RemoteFailure, RemoteDischargeSmoothPar)`

    Used auxiliary method:
      |smooth_logistic1|

    Examples:

        As in the examples above, define a short simulation time period first:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'

        Prepare the dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> derived.toy.update()

        Define a safety factor of 0.5 m³/s for the summer months and
        no safety factor at all for the winter months:

        >>> remotedischargesafety(_11_1_12=0.0, _03_31_12=0.0,
        ...                       _04_1_12=1.0, _10_31_12=1.0)
        >>> derived.remotedischargesmoothpar.update()

        Assume the actual demand at the cross section downsstream has actually
        been estimated to be 2 m³/s:

        >>> fluxes.remotedemand = 2.0

        Prepare a test function, that calculates the required discharge
        based on the parameter values defined above and for a "remote
        failure" values ranging between -4 and 4 m³/s:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_requiredremoterelease_v1,
        ...                 last_example=9,
        ...                 parseqs=(fluxes.remotefailure,
        ...                          fluxes.requiredremoterelease))
        >>> test.nexts.remotefailure = range(-4, 5)

        On May 31, the safety factor is 0 m³/s.  Hence no discharge is
        added to the estimated remote demand of 2 m³/s:

        >>> model.idx_sim = pub.timegrids.init['2001.03.31']
        >>> test()
        | ex. | remotefailure | requiredremoterelease |
        -----------------------------------------------
        |   1 |          -4.0 |                   2.0 |
        |   2 |          -3.0 |                   2.0 |
        |   3 |          -2.0 |                   2.0 |
        |   4 |          -1.0 |                   2.0 |
        |   5 |           0.0 |                   2.0 |
        |   6 |           1.0 |                   2.0 |
        |   7 |           2.0 |                   2.0 |
        |   8 |           3.0 |                   2.0 |
        |   9 |           4.0 |                   2.0 |

        On April 1, the safety factor is 1 m³/s.  If the remote failure was
        exactly zero in the past, meaning the control of the dam was perfect,
        only 0.5 m³/s are added to the estimated remote demand of 2 m³/s.
        If the actual recharge did actually fall below the threshold value,
        up to 1 m³/s is added. If the the actual discharge exceeded the
        threshold value by 2 or 3 m³/s, virtually nothing is added:

        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> test()
        | ex. | remotefailure | requiredremoterelease |
        -----------------------------------------------
        |   1 |          -4.0 |                   2.0 |
        |   2 |          -3.0 |              2.000001 |
        |   3 |          -2.0 |              2.000102 |
        |   4 |          -1.0 |                  2.01 |
        |   5 |           0.0 |                   2.5 |
        |   6 |           1.0 |                  2.99 |
        |   7 |           2.0 |              2.999898 |
        |   8 |           3.0 |              2.999999 |
        |   9 |           4.0 |                   3.0 |

    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.requiredremoterelease = (
        flu.remotedemand+con.remotedischargesafety[der.toy[self.idx_sim]] *
        smoothutils.smooth_logistic1(
            flu.remotefailure,
            der.remotedischargesmoothpar[der.toy[self.idx_sim]]))