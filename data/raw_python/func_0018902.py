def calc_sbes_v1(self):
    """Calculate the frozen part of stand precipitation.

    Required control parameters:
      |NHRU|
      |TGr|
      |TSp|

    Required flux sequences:
      |TKor|
      |NBes|

    Calculated flux sequence:
      |SBes|

    Examples:

        In the first example, the threshold temperature of seven hydrological
        response units is 0 °C and the corresponding temperature interval of
        mixed precipitation 2 °C:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(7)
        >>> tgr(0.0)
        >>> tsp(2.0)

        The value of |NBes| is zero above 1 °C and equal to the value of
        |NBes| below -1 °C.  Between these temperature values, |NBes|
        decreases linearly:

        >>> fluxes.nbes = 4.0
        >>> fluxes.tkor = -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0
        >>> model.calc_sbes_v1()
        >>> fluxes.sbes
        sbes(4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0)

        Note the special case of a zero temperature interval.  With the
        actual temperature being equal to the threshold temperature, the
        the value of `sbes` is zero:

        >>> tsp(0.)
        >>> model.calc_sbes_v1()
        >>> fluxes.sbes
        sbes(4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        if flu.nbes[k] <= 0.:
            flu.sbes[k] = 0.
        elif flu.tkor[k] >= (con.tgr[k]+con.tsp[k]/2.):
            flu.sbes[k] = 0.
        elif flu.tkor[k] <= (con.tgr[k]-con.tsp[k]/2.):
            flu.sbes[k] = flu.nbes[k]
        else:
            flu.sbes[k] = ((((con.tgr[k]+con.tsp[k]/2.)-flu.tkor[k]) /
                            con.tsp[k])*flu.nbes[k])