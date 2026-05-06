def calc_fracrain_v1(self):
    """Determine the temperature-dependent fraction of (liquid) rainfall
    and (total) precipitation.

    Required control parameters:
      |NmbZones|
      |TT|,
      |TTInt|

    Required flux sequence:
      |TC|

    Calculated flux sequences:
      |FracRain|

    Basic equation:
      :math:`FracRain = \\frac{TC-(TT-\\frac{TTInt}{2})}{TTInt}`

    Restriction:
      :math:`0 \\leq FracRain \\leq 1`

    Examples:

        The threshold temperature of seven zones is 0°C and the corresponding
        temperature interval of mixed precipitation 2°C:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(7)
        >>> tt(0.0)
        >>> ttint(2.0)

        The fraction of rainfall is zero below -1°C, is one above 1°C and
        increases linearly in between:

        >>> fluxes.tc = -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0
        >>> model.calc_fracrain_v1()
        >>> fluxes.fracrain
        fracrain(0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0)

        Note the special case of a zero temperature interval.  With a
        actual temperature being equal to the threshold temperature, the
        rainfall fraction is one:

        >>> ttint(0.0)
        >>> model.calc_fracrain_v1()
        >>> fluxes.fracrain
        fracrain(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        if flu.tc[k] >= (con.tt[k]+con.ttint[k]/2.):
            flu.fracrain[k] = 1.
        elif flu.tc[k] <= (con.tt[k]-con.ttint[k]/2.):
            flu.fracrain[k] = 0.
        else:
            flu.fracrain[k] = ((flu.tc[k]-(con.tt[k]-con.ttint[k]/2.)) /
                               con.ttint[k])