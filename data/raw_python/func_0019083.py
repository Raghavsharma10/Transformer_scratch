def calc_inuh_v1(self):
    """Calculate the unit hydrograph input.

    Required derived parameters:
      |RelLandArea|

    Required flux sequences:
      |Q0|
      |Q1|

    Calculated flux sequence:
      |InUH|

    Basic equation:
        :math:`InUH = Q0 + Q1`

    Example:

        The unit hydrographs receives base flow from the whole subbasin
        and direct flow from zones of type field, forest and glacier only.
        In the following example, these occupy only one half of the
        subbasin, which is why the partial input of q0 is halved:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> derived.rellandarea = 0.5
        >>> fluxes.q0 = 4.0
        >>> fluxes.q1 = 1.0
        >>> model.calc_inuh_v1()
        >>> fluxes.inuh
        inuh(3.0)

    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.inuh = der.rellandarea*flu.q0+flu.q1