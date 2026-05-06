def calc_actualremoterelieve_v1(self):
    """Calculate the actual amount of water released to a remote location
    to relieve the dam during high flow conditions.

    Required control parameter:
      |RemoteRelieveTolerance|

    Required flux sequences:
      |AllowedRemoteRelieve|
      |PossibleRemoteRelieve|

    Calculated flux sequence:
      |ActualRemoteRelieve|

    Basic equation - discontinous:
      :math:`ActualRemoteRelease = min(PossibleRemoteRelease,
      AllowedRemoteRelease)`

    Basic equation - continous:
      :math:`ActualRemoteRelease = smooth_min1(PossibleRemoteRelease,
      AllowedRemoteRelease, RemoteRelieveTolerance)`

    Used auxiliary methods:
      |smooth_min1|
      |smooth_max1|


    Note that the given continous basic equation is a simplification of
    the complete algorithm to calculate |ActualRemoteRelieve|, which also
    makes use of |smooth_max1| to prevent from gaining negative values
    in a smooth manner.


    Examples:

        Prepare a dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()

        Prepare a test function object that performs seven examples with
        |PossibleRemoteRelieve| ranging from -1 to 5 m³/s:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_actualremoterelieve_v1,
        ...                 last_example=7,
        ...                 parseqs=(fluxes.possibleremoterelieve,
        ...                          fluxes.actualremoterelieve))
        >>> test.nexts.possibleremoterelieve = range(-1, 6)

        We begin with a |AllowedRemoteRelieve| value of 3 m³/s:

        >>> fluxes.allowedremoterelieve = 3.0

        Through setting the value of |RemoteRelieveTolerance| to the
        lowest possible value, there is no smoothing.  Instead, the
        relationship between |ActualRemoteRelieve| and |PossibleRemoteRelieve|
        follows the simple discontinous minimum function:

        >>> remoterelievetolerance(0.0)
        >>> test()
        | ex. | possibleremoterelieve | actualremoterelieve |
        -----------------------------------------------------
        |   1 |                  -1.0 |                 0.0 |
        |   2 |                   0.0 |                 0.0 |
        |   3 |                   1.0 |                 1.0 |
        |   4 |                   2.0 |                 2.0 |
        |   5 |                   3.0 |                 3.0 |
        |   6 |                   4.0 |                 3.0 |
        |   7 |                   5.0 |                 3.0 |

        Increasing the value of parameter |RemoteRelieveTolerance| to a
        sensible value results in a moderate smoothing:

        >>> remoterelievetolerance(0.2)
        >>> test()
        | ex. | possibleremoterelieve | actualremoterelieve |
        -----------------------------------------------------
        |   1 |                  -1.0 |                 0.0 |
        |   2 |                   0.0 |                 0.0 |
        |   3 |                   1.0 |            0.970639 |
        |   4 |                   2.0 |             1.89588 |
        |   5 |                   3.0 |            2.584112 |
        |   6 |                   4.0 |            2.896195 |
        |   7 |                   5.0 |            2.978969 |

        Even when setting a very large smoothing parameter value, the actual
        remote relieve does not fall below 0 m³/s:

        >>> remoterelievetolerance(1.0)
        >>> test()
        | ex. | possibleremoterelieve | actualremoterelieve |
        -----------------------------------------------------
        |   1 |                  -1.0 |                 0.0 |
        |   2 |                   0.0 |                 0.0 |
        |   3 |                   1.0 |            0.306192 |
        |   4 |                   2.0 |            0.634882 |
        |   5 |                   3.0 |            1.037708 |
        |   6 |                   4.0 |            1.436494 |
        |   7 |                   5.0 |            1.788158 |

        Now we repeat the last example with a allowed remote relieve of
        only 0.03 m³/s instead of 3 m³/s:

        >>> fluxes.allowedremoterelieve = 0.03
        >>> test()
        | ex. | possibleremoterelieve | actualremoterelieve |
        -----------------------------------------------------
        |   1 |                  -1.0 |                 0.0 |
        |   2 |                   0.0 |                 0.0 |
        |   3 |                   1.0 |                0.03 |
        |   4 |                   2.0 |                0.03 |
        |   5 |                   3.0 |                0.03 |
        |   6 |                   4.0 |                0.03 |
        |   7 |                   5.0 |                0.03 |

        The result above is as expected, but the smooth part of the
        relationship is not resolved.  By increasing the resolution we
        see a relationship that corresponds to the one shown above
        for an allowed relieve of 3 m³/s.  This points out, that the
        degree of smoothing is releative to the allowed relieve:

        >>> import numpy
        >>> test.nexts.possibleremoterelieve = numpy.arange(-0.01, 0.06, 0.01)
        >>> test()
        | ex. | possibleremoterelieve | actualremoterelieve |
        -----------------------------------------------------
        |   1 |                 -0.01 |                 0.0 |
        |   2 |                   0.0 |                 0.0 |
        |   3 |                  0.01 |            0.003062 |
        |   4 |                  0.02 |            0.006349 |
        |   5 |                  0.03 |            0.010377 |
        |   6 |                  0.04 |            0.014365 |
        |   7 |                  0.05 |            0.017882 |

        One can reperform the shown experiments with an even higher
        resolution to see that the relationship between
        |ActualRemoteRelieve| and |PossibleRemoteRelieve| is
        (at least in most cases) in fact very smooth.  But a more analytical
        approach would possibly be favourable regarding the smoothness in
        some edge cases and computational efficiency.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    d_smoothpar = con.remoterelievetolerance*flu.allowedremoterelieve
    flu.actualremoterelieve = smoothutils.smooth_min1(
        flu.possibleremoterelieve, flu.allowedremoterelieve, d_smoothpar)
    for dummy in range(5):
        d_smoothpar /= 5.
        flu.actualremoterelieve = smoothutils.smooth_max1(
            flu.actualremoterelieve, 0., d_smoothpar)
        d_smoothpar /= 5.
        flu.actualremoterelieve = smoothutils.smooth_min1(
            flu.actualremoterelieve, flu.possibleremoterelieve, d_smoothpar)
    flu.actualremoterelieve = min(flu.actualremoterelieve,
                                  flu.possibleremoterelieve)
    flu.actualremoterelieve = min(flu.actualremoterelieve,
                                  flu.allowedremoterelieve)
    flu.actualremoterelieve = max(flu.actualremoterelieve, 0.)