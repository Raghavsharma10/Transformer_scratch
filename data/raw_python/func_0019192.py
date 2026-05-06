def calc_allowedremoterelieve_v2(self):
    """Calculate the allowed maximum relieve another location
    is allowed to discharge into the dam.

    Required control parameters:
      |HighestRemoteRelieve|
      |WaterLevelRelieveThreshold|

    Required derived parameter:
      |WaterLevelRelieveSmoothPar|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |AllowedRemoteRelieve|

    Basic equation:
      :math:`ActualRemoteRelease = HighestRemoteRelease \\cdot
      smooth_{logistic1}(WaterLevelRelieveThreshold-WaterLevel,
      WaterLevelRelieveSmoothPar)`

    Used auxiliary method:
      |smooth_logistic1|

    Examples:

        All control parameters that are involved in the calculation of
        |AllowedRemoteRelieve| are derived from |SeasonalParameter|.
        This allows to simulate seasonal dam control schemes.
        To show how this works, we first define a short simulation
        time period of only two days:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'

        Now we prepare the dam model and define two different control
        schemes for the hydrological summer (April to October) and
        winter month (November to May)

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> highestremoterelieve(_11_1_12=1.0, _03_31_12=1.0,
        ...                      _04_1_12=2.0, _10_31_12=2.0)
        >>> waterlevelrelievethreshold(_11_1_12=3.0, _03_31_12=2.0,
        ...                            _04_1_12=4.0, _10_31_12=4.0)
        >>> waterlevelrelievetolerance(_11_1_12=0.0, _03_31_12=0.0,
        ...                            _04_1_12=1.0, _10_31_12=1.0)
        >>> derived.waterlevelrelievesmoothpar.update()
        >>> derived.toy.update()

        The following test function is supposed to calculate
        |AllowedRemoteRelieve| for values of |WaterLevel| ranging
        from 0 and 8 m:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model,
        ...                 model.calc_allowedremoterelieve_v2,
        ...                 last_example=9,
        ...                 parseqs=(aides.waterlevel,
        ...                          fluxes.allowedremoterelieve))
        >>> test.nexts.waterlevel = range(9)

        On March 30 (which is the last day of the winter month and the
        first day of the simulation period), the value of
        |WaterLevelRelieveSmoothPar| is zero.  Hence, |AllowedRemoteRelieve|
        drops abruptly from 1 m³/s (the value of |HighestRemoteRelieve|) to
        0 m³/s, as soon as |WaterLevel| reaches 3 m (the value
        of |WaterLevelRelieveThreshold|):

        >>> model.idx_sim = pub.timegrids.init['2001.03.30']
        >>> test(first_example=2, last_example=6)
        | ex. | waterlevel | allowedremoterelieve |
        -------------------------------------------
        |   3 |        1.0 |                  1.0 |
        |   4 |        2.0 |                  1.0 |
        |   5 |        3.0 |                  0.0 |
        |   6 |        4.0 |                  0.0 |

        On April 1 (which is the first day of the sommer month and the
        last day of the simulation period), all parameter values are
        increased.  The value of parameter |WaterLevelRelieveSmoothPar|
        is 1 m.  Hence, loosely speaking, |AllowedRemoteRelieve| approaches
        the "discontinuous extremes (2 m³/s -- which is the value of
        |HighestRemoteRelieve| -- and 0 m³/s) to 99 % within a span of
        2 m³/s around the original threshold value of 4 m³/s defined by
        |WaterLevelRelieveThreshold|:

        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> test()
        | ex. | waterlevel | allowedremoterelieve |
        -------------------------------------------
        |   1 |        0.0 |                  2.0 |
        |   2 |        1.0 |             1.999998 |
        |   3 |        2.0 |             1.999796 |
        |   4 |        3.0 |                 1.98 |
        |   5 |        4.0 |                  1.0 |
        |   6 |        5.0 |                 0.02 |
        |   7 |        6.0 |             0.000204 |
        |   8 |        7.0 |             0.000002 |
        |   9 |        8.0 |                  0.0 |
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    toy = der.toy[self.idx_sim]
    flu.allowedremoterelieve = (
        con.highestremoterelieve[toy] *
        smoothutils.smooth_logistic1(
            con.waterlevelrelievethreshold[toy]-aid.waterlevel,
            der.waterlevelrelievesmoothpar[toy]))