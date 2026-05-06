def calc_ep_v1(self):
    """Adjust potential norm evaporation to the actual temperature.

    Required control parameters:
      |NmbZones|
      |ETF|

    Required input sequence:
      |EPN|
      |TN|

    Required flux sequence:
      |TMean|

    Calculated flux sequences:
      |EP|

    Basic equation:
      :math:`EP = EPN \\cdot (1 + ETF \\cdot (TMean - TN))`

    Restriction:
      :math:`0 \\leq EP \\leq 2 \\cdot EPN`

    Examples:

        Assume four zones with different values of the temperature
        related factor for the adjustment of evaporation (the
        negative value of the first zone is not meaningful, but used
        for illustration purporses):

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(4)
        >>> etf(-0.5, 0.0, 0.1, 0.5)
        >>> inputs.tn = 20.0
        >>> inputs.epn = 2.0

        With mean temperature equal to norm temperature, actual
        (uncorrected) evaporation is equal to norm evaporation:

        >>> fluxes.tmean = 20.0
        >>> model.calc_ep_v1()
        >>> fluxes.ep
        ep(2.0, 2.0, 2.0, 2.0)

        With mean temperature 5°C higher than norm temperature, potential
        evaporation is increased by 1 mm for the third zone, which
        possesses a very common adjustment factor.  For the first zone,
        potential evaporation is 0 mm (which is the smallest value
        allowed), and for the fourth zone it is the double value of the
        norm evaporation (which is the largest value allowed):

        >>> fluxes.tmean  = 25.0
        >>> model.calc_ep_v1()
        >>> fluxes.ep
        ep(0.0, 2.0, 3.0, 4.0)
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        flu.ep[k] = inp.epn*(1.+con.etf[k]*(flu.tmean-inp.tn))
        flu.ep[k] = min(max(flu.ep[k], 0.), 2.*inp.epn)