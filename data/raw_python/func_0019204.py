def calc_actualrelease_v1(self):
    """Calculate the actual water release that can be supplied by the
    dam considering the targeted release and the given water level.

    Required control parameter:
      |WaterLevelMinimumThreshold|

    Required derived parameters:
      |WaterLevelMinimumSmoothPar|

    Required flux sequence:
      |TargetedRelease|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |ActualRelease|

    Basic equation:
      :math:`ActualRelease = TargetedRelease \\cdot
      smooth_{logistic1}(WaterLevelMinimumThreshold-WaterLevel,
      WaterLevelMinimumSmoothPar)`

    Used auxiliary method:
      |smooth_logistic1|

    Examples:

        Prepare the dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()

        Assume the required release has previously been estimated
        to be 2 m³/s:

        >>> fluxes.targetedrelease = 2.0

        Prepare a test function, that calculates the targeted water release
        for water levels ranging between -1 and 5 m:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_actualrelease_v1,
        ...                 last_example=7,
        ...                 parseqs=(aides.waterlevel,
        ...                          fluxes.actualrelease))
        >>> test.nexts.waterlevel = range(-1, 6)

        .. _dam_calc_actualrelease_v1_ex01:

        **Example 1**

        Firstly, we define a sharp minimum water level of 0 m:

        >>> waterlevelminimumthreshold(0.)
        >>> waterlevelminimumtolerance(0.)
        >>> derived.waterlevelminimumsmoothpar.update()

        The following test results show that the water releae is reduced
        to 0 m³/s for water levels (even slightly) lower than 0 m and is
        identical with the required value of 2 m³/s (even slighlty) above 0 m:

        >>> test()
        | ex. | waterlevel | actualrelease |
        ------------------------------------
        |   1 |       -1.0 |           0.0 |
        |   2 |        0.0 |           1.0 |
        |   3 |        1.0 |           2.0 |
        |   4 |        2.0 |           2.0 |
        |   5 |        3.0 |           2.0 |
        |   6 |        4.0 |           2.0 |
        |   7 |        5.0 |           2.0 |


        One may have noted that in the above example the calculated water
        release is 1 m³/s (which is exactly the half of the targeted release)
        at a water level of 1 m.  This looks suspiciously lake a flaw but
        is not of any importance considering the fact, that numerical
        integration algorithms will approximate the analytical solution
        of a complete emptying of a dam emtying (which is a water level
        of 0 m), only with a certain accuracy.

        .. _dam_calc_actualrelease_v1_ex02:

        **Example 2**

        Nonetheless, it can (besides some other possible advantages)
        dramatically increase the speed of numerical integration algorithms
        to define a smooth transition area instead of sharp threshold value,
        like in the following example:

        >>> waterlevelminimumthreshold(4.)
        >>> waterlevelminimumtolerance(1.)
        >>> derived.waterlevelminimumsmoothpar.update()

        Now, 98 % of the variation of the total range from 0 m³/s to 2 m³/s
        occurs between a water level of 3 m and 5 m:

        >>> test()
        | ex. | waterlevel | actualrelease |
        ------------------------------------
        |   1 |       -1.0 |           0.0 |
        |   2 |        0.0 |           0.0 |
        |   3 |        1.0 |      0.000002 |
        |   4 |        2.0 |      0.000204 |
        |   5 |        3.0 |          0.02 |
        |   6 |        4.0 |           1.0 |
        |   7 |        5.0 |          1.98 |

        .. _dam_calc_actualrelease_v1_ex03:

        **Example 3**

        Note that it is possible to set both parameters in a manner that
        might result in negative water stages beyond numerical inaccuracy:

        >>> waterlevelminimumthreshold(1.)
        >>> waterlevelminimumtolerance(2.)
        >>> derived.waterlevelminimumsmoothpar.update()

        Here, the actual water release is 0.18 m³/s for a water level
        of 0 m.  Hence water stages in the range of 0 m to -1 m or
        even -2 m might occur during the simulation of long drought events:

        >>> test()
        | ex. | waterlevel | actualrelease |
        ------------------------------------
        |   1 |       -1.0 |          0.02 |
        |   2 |        0.0 |       0.18265 |
        |   3 |        1.0 |           1.0 |
        |   4 |        2.0 |       1.81735 |
        |   5 |        3.0 |          1.98 |
        |   6 |        4.0 |      1.997972 |
        |   7 |        5.0 |      1.999796 |

    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    flu.actualrelease = (flu.targetedrelease *
                         smoothutils.smooth_logistic1(
                             aid.waterlevel-con.waterlevelminimumthreshold,
                             der.waterlevelminimumsmoothpar))