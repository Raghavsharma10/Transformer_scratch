def calc_requiredremotesupply_v1(self):
    """Calculate the required maximum supply from another location
    that can be discharged into the dam.

    Required control parameters:
      |HighestRemoteSupply|
      |WaterLevelSupplyThreshold|

    Required derived parameter:
      |WaterLevelSupplySmoothPar|

    Required aide sequence:
      |WaterLevel|

    Calculated flux sequence:
      |RequiredRemoteSupply|

    Basic equation:
      :math:`RequiredRemoteSupply = HighestRemoteSupply \\cdot
      smooth_{logistic1}(WaterLevelSupplyThreshold-WaterLevel,
      WaterLevelSupplySmoothPar)`

    Used auxiliary method:
      |smooth_logistic1|

    Examples:

        Method |calc_requiredremotesupply_v1| is functionally identical
        with method |calc_allowedremoterelieve_v2|.  Hence the following
        examples serve for testing purposes only (see the documentation
        on function |calc_allowedremoterelieve_v2| for more detailed
        information):

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'
        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> highestremotesupply(_11_1_12=1.0, _03_31_12=1.0,
        ...                     _04_1_12=2.0, _10_31_12=2.0)
        >>> waterlevelsupplythreshold(_11_1_12=3.0, _03_31_12=2.0,
        ...                           _04_1_12=4.0, _10_31_12=4.0)
        >>> waterlevelsupplytolerance(_11_1_12=0.0, _03_31_12=0.0,
        ...                           _04_1_12=1.0, _10_31_12=1.0)
        >>> derived.waterlevelsupplysmoothpar.update()
        >>> derived.toy.update()
        >>> from hydpy import UnitTest
        >>> test = UnitTest(model,
        ...                 model.calc_requiredremotesupply_v1,
        ...                 last_example=9,
        ...                 parseqs=(aides.waterlevel,
        ...                          fluxes.requiredremotesupply))
        >>> test.nexts.waterlevel = range(9)
        >>> model.idx_sim = pub.timegrids.init['2001.03.30']
        >>> test(first_example=2, last_example=6)
        | ex. | waterlevel | requiredremotesupply |
        -------------------------------------------
        |   3 |        1.0 |                  1.0 |
        |   4 |        2.0 |                  1.0 |
        |   5 |        3.0 |                  0.0 |
        |   6 |        4.0 |                  0.0 |
        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> test()
        | ex. | waterlevel | requiredremotesupply |
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
    flu.requiredremotesupply = (
        con.highestremotesupply[toy] *
        smoothutils.smooth_logistic1(
            con.waterlevelsupplythreshold[toy]-aid.waterlevel,
            der.waterlevelsupplysmoothpar[toy]))