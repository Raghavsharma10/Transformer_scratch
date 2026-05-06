def calc_pc_v1(self):
    """Apply the precipitation correction factors and adjust precipitation
    to the altitude of the individual zones.

    Required control parameters:
      |NmbZones|
      |PCorr|
      |PCAlt|
      |ZoneZ|
      |ZRelP|

    Required input sequence:
      |P|

    Required flux sequences:
      |RfC|
      |SfC|

    Calculated flux sequences:
      |PC|

    Basic equation:
      :math:`PC = P \\cdot PCorr
      \\cdot (1+PCAlt \\cdot (ZoneZ-ZRelP))
      \\cdot (RfC + SfC)`

    Examples:

        Five zones are at an elevation of 200 m.  A precipitation value
        of 5 mm has been measured at a gauge at an elevation of 300 m:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(5)
        >>> zrelp(2.0)
        >>> zonez(3.0)
        >>> inputs.p = 5.0

        The first four zones illustrate the individual precipitation
        corrections due to the general precipitation correction factor
        (|PCorr|, first zone), the altitude correction factor (|PCAlt|,
        second zone), the rainfall related correction (|RfC|, third zone),
        and the snowfall related correction factor (|SfC|, fourth zone).
        The fifth zone illustrates the interaction between all corrections:

        >>> pcorr(1.3, 1.0, 1.0, 1.0, 1.3)
        >>> pcalt(0.0, 0.1, 0.0, 0.0, 0.1)
        >>> fluxes.rfc = 0.5, 0.5, 0.4, 0.5, 0.4
        >>> fluxes.sfc = 0.5, 0.5, 0.5, 0.7, 0.7
        >>> model.calc_pc_v1()
        >>> fluxes.pc
        pc(6.5, 5.5, 4.5, 6.0, 7.865)

        Usually, one would set zero or positive values for parameter |PCAlt|.
        But it is also allowed to set negative values, in order to reflect
        possible negative relationships between precipitation and altitude.
        To prevent from calculating negative precipitation when too large
        negative values are applied, a truncation is performed:

        >>> pcalt(-1.0)
        >>> model.calc_pc_v1()
        >>> fluxes.pc
        pc(0.0, 0.0, 0.0, 0.0, 0.0)

    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        flu.pc[k] = inp.p*(1.+con.pcalt[k]*(con.zonez[k]-con.zrelp))
        if flu.pc[k] <= 0.:
            flu.pc[k] = 0.
        else:
            flu.pc[k] *= con.pcorr[k]*(flu.rfc[k]+flu.sfc[k])