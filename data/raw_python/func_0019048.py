def calc_rk_v1(self):
    """Determine the actual traveling time of the water (not of the wave!).

    Required derived parameter:
      |Sek|

    Required flux sequences:
      |AG|
      |QRef|

    Calculated flux sequence:
      |RK|

    Basic equation:
      :math:`RK = \\frac{Laen \\cdot A}{QRef}`

    Examples:

        First, note that the traveling time is determined in the unit of the
        actual simulation step size:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> laen(25.0)
        >>> derived.sek(24*60*60)
        >>> fluxes.ag = 10.0
        >>> fluxes.qref = 1.0
        >>> model.calc_rk_v1()
        >>> fluxes.rk
        rk(2.893519)

        Second, for negative values or zero values of |AG| or |QRef|,
        the value of |RK| is set to zero:

        >>> fluxes.ag = 0.0
        >>> fluxes.qref = 1.0
        >>> model.calc_rk_v1()
        >>> fluxes.rk
        rk(0.0)

        >>> fluxes.ag = 0.0
        >>> fluxes.qref = 1.0
        >>> model.calc_rk_v1()
        >>> fluxes.rk
        rk(0.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    if (flu.ag > 0.) and (flu.qref > 0.):
        flu.rk = (1000.*con.laen*flu.ag)/(der.sek*flu.qref)
    else:
        flu.rk = 0.