def calc_inuz_v1(self):
    """Accumulate the total inflow into the upper zone layer.

    Required control parameters:
      |NmbZones|
      |ZoneType|

    Required derived parameters:
      |RelLandZoneArea|

    Required fluxes sequences:
      |R|
      |CF|

    Calculated flux sequence:
      |InUZ|

    Basic equation:
      :math:`InUZ = R - CF`

    Examples:

        Initialize three zones of different relative `land sizes`
        (area related to the total size of the subbasin except lake areas):

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(3)
        >>> zonetype(FIELD, ILAKE, GLACIER)
        >>> derived.rellandzonearea = 2.0/3.0, 0.0, 1.0/3.0
        >>> fluxes.r = 6.0, 0.0, 2.0
        >>> fluxes.cf = 2.0, 0.0, 1.0
        >>> model.calc_inuz_v1()
        >>> fluxes.inuz
        inuz(3.0)

        Internal lakes do not contribute to the upper zone layer.  Hence
        for a subbasin consisting only of interal lakes a zero input
        value would be calculated:

        >>> zonetype(ILAKE, ILAKE, ILAKE)
        >>> model.calc_inuz_v1()
        >>> fluxes.inuz
        inuz(0.0)

    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.inuz = 0.
    for k in range(con.nmbzones):
        if con.zonetype[k] != ILAKE:
            flu.inuz += der.rellandzonearea[k]*(flu.r[k]-flu.cf[k])