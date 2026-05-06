def update_actualremoterelieve_v1(self):
    """Constrain the actual relieve discharge to a remote location.

    Required control parameter:
      |HighestRemoteDischarge|

    Required derived parameter:
      |HighestRemoteSmoothPar|

    Updated flux sequence:
      |ActualRemoteRelieve|

    Basic equation - discontinous:
      :math:`ActualRemoteRelieve = min(ActualRemoteRelease,
      HighestRemoteDischarge)`

    Basic equation - continous:
      :math:`ActualRemoteRelieve = smooth_min1(ActualRemoteRelieve,
      HighestRemoteDischarge, HighestRemoteSmoothPar)`

    Used auxiliary methods:
      |smooth_min1|
      |smooth_max1|

    Note that the given continous basic equation is a simplification of
    the complete algorithm to update |ActualRemoteRelieve|, which also
    makes use of |smooth_max1| to prevent from gaining negative values
    in a smooth manner.

    Examples:

        Prepare a dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()

        Prepare a test function object that performs eight examples with
        |ActualRemoteRelieve| ranging from 0 to 8 m³/s and a fixed
        initial value of parameter |HighestRemoteDischarge| of 4 m³/s:

        >>> highestremotedischarge(4.0)
        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.update_actualremoterelieve_v1,
        ...                 last_example=8,
        ...                 parseqs=(fluxes.actualremoterelieve,))
        >>> test.nexts.actualremoterelieve = range(8)

        Through setting the value of |HighestRemoteTolerance| to the
        lowest possible value, there is no smoothing.  Instead, the
        shown relationship agrees with a combination of the discontinuous
        minimum and maximum function:

        >>> highestremotetolerance(0.0)
        >>> derived.highestremotesmoothpar.update()
        >>> test()
        | ex. | actualremoterelieve |
        -----------------------------
        |   1 |                 0.0 |
        |   2 |                 1.0 |
        |   3 |                 2.0 |
        |   4 |                 3.0 |
        |   5 |                 4.0 |
        |   6 |                 4.0 |
        |   7 |                 4.0 |
        |   8 |                 4.0 |

        Setting a sensible |HighestRemoteTolerance| value results in
        a moderate smoothing:

        >>> highestremotetolerance(0.1)
        >>> derived.highestremotesmoothpar.update()
        >>> test()
        | ex. | actualremoterelieve |
        -----------------------------
        |   1 |                 0.0 |
        |   2 |            0.999999 |
        |   3 |             1.99995 |
        |   4 |            2.996577 |
        |   5 |            3.836069 |
        |   6 |            3.991578 |
        |   7 |            3.993418 |
        |   8 |            3.993442 |

        Method |update_actualremoterelieve_v1| is defined in a similar
        way as method |calc_actualremoterelieve_v1|.  Please read the
        documentation on |calc_actualremoterelieve_v1| for further
        information.
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    d_smooth = der.highestremotesmoothpar
    d_highest = con.highestremotedischarge
    d_value = smoothutils.smooth_min1(
        flu.actualremoterelieve, d_highest, d_smooth)
    for dummy in range(5):
        d_smooth /= 5.
        d_value = smoothutils.smooth_max1(
            d_value, 0., d_smooth)
        d_smooth /= 5.
        d_value = smoothutils.smooth_min1(
            d_value, d_highest, d_smooth)
    d_value = min(d_value, flu.actualremoterelieve)
    d_value = min(d_value, d_highest)
    flu.actualremoterelieve = max(d_value, 0.)