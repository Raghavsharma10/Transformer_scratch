def calc_actualremoterelease_v1(self):
    """Calculate the actual remote water release that can be supplied by the
    dam considering the required remote release and the given water level.

    Required control parameter:
      |WaterLevelMinimumRemoteThreshold|

    Required derived parameters:
      |WaterLevelMinimumRemoteSmoothPar|

    Required flux sequence:
      |RequiredRemoteRelease|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |ActualRemoteRelease|

    Basic equation:
      :math:`ActualRemoteRelease = RequiredRemoteRelease \\cdot
      smooth_{logistic1}(WaterLevelMinimumRemoteThreshold-WaterLevel,
      WaterLevelMinimumRemoteSmoothPar)`

    Used auxiliary method:
      |smooth_logistic1|

    Examples:

        Note that method |calc_actualremoterelease_v1| is functionally
        identical with method |calc_actualrelease_v1|.  This is why we
        omit to explain the following examples, as they are just repetitions
        of the ones of method |calc_actualremoterelease_v1| with partly
        different variable names.  Please follow the links to read the
        corresponding explanations.

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> fluxes.requiredremoterelease = 2.0
        >>> from hydpy import UnitTest
        >>> test = UnitTest(model, model.calc_actualremoterelease_v1,
        ...                 last_example=7,
        ...                 parseqs=(aides.waterlevel,
        ...                          fluxes.actualremoterelease))
        >>> test.nexts.waterlevel = range(-1, 6)

        :ref:`Recalculation of example 1 <dam_calc_actualrelease_v1_ex01>`

        >>> waterlevelminimumremotethreshold(0.)
        >>> waterlevelminimumremotetolerance(0.)
        >>> derived.waterlevelminimumremotesmoothpar.update()
        >>> test()
        | ex. | waterlevel | actualremoterelease |
        ------------------------------------------
        |   1 |       -1.0 |                 0.0 |
        |   2 |        0.0 |                 1.0 |
        |   3 |        1.0 |                 2.0 |
        |   4 |        2.0 |                 2.0 |
        |   5 |        3.0 |                 2.0 |
        |   6 |        4.0 |                 2.0 |
        |   7 |        5.0 |                 2.0 |


        :ref:`Recalculation of example 2 <dam_calc_actualrelease_v1_ex02>`

        >>> waterlevelminimumremotethreshold(4.)
        >>> waterlevelminimumremotetolerance(1.)
        >>> derived.waterlevelminimumremotesmoothpar.update()
        >>> test()
        | ex. | waterlevel | actualremoterelease |
        ------------------------------------------
        |   1 |       -1.0 |                 0.0 |
        |   2 |        0.0 |                 0.0 |
        |   3 |        1.0 |            0.000002 |
        |   4 |        2.0 |            0.000204 |
        |   5 |        3.0 |                0.02 |
        |   6 |        4.0 |                 1.0 |
        |   7 |        5.0 |                1.98 |

        :ref:`Recalculation of example 3 <dam_calc_actualrelease_v1_ex03>`

        >>> waterlevelminimumremotethreshold(1.)
        >>> waterlevelminimumremotetolerance(2.)
        >>> derived.waterlevelminimumremotesmoothpar.update()
        >>> test()
        | ex. | waterlevel | actualremoterelease |
        ------------------------------------------
        |   1 |       -1.0 |                0.02 |
        |   2 |        0.0 |             0.18265 |
        |   3 |        1.0 |                 1.0 |
        |   4 |        2.0 |             1.81735 |
        |   5 |        3.0 |                1.98 |
        |   6 |        4.0 |            1.997972 |
        |   7 |        5.0 |            1.999796 |
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    flu.actualremoterelease = (
        flu.requiredremoterelease *
        smoothutils.smooth_logistic1(
            aid.waterlevel-con.waterlevelminimumremotethreshold,
            der.waterlevelminimumremotesmoothpar))