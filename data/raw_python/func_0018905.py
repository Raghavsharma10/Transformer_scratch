def calc_wada_waes_v1(self):
    """Calculate the actual water release from the snow cover.

    Required control parameters:
      |NHRU|
      |Lnk|
      |PWMax|

    Required flux sequences:
      |NBes|

    Calculated flux sequence:
      |WaDa|

    Updated state sequence:
      |WAeS|

    Basic equations:
      :math:`\\frac{dWAeS}{dt} = NBes - WaDa`
      :math:`WAeS \\leq PWMax \\cdot WATS`

    Examples:

        For simplicity, the threshold parameter |PWMax| is set to a value
        of two for each of the six initialized HRUs.  Thus, snow cover can
        hold as much liquid water as it contains frozen water.  Stand
        precipitation is also always set to the same value, but the initial
        conditions of the snow cover are varied:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(6)
        >>> lnk(FLUSS, SEE, ACKER, ACKER, ACKER, ACKER)
        >>> pwmax(2.0)
        >>> fluxes.nbes = 1.0
        >>> states.wats = 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
        >>> states.waes = 1.0, 1.0, 0.0, 1.0, 1.5, 2.0
        >>> model.calc_wada_waes_v1()
        >>> states.waes
        waes(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
        >>> fluxes.wada
        wada(1.0, 1.0, 1.0, 0.0, 0.5, 1.0)

        Note the special cases of the first two HRUs of type |FLUSS| and
        |SEE|.  For water areas, stand precipitaton |NBes| is generally
        passed to |WaDa| and |WAeS| is set to zero.  For all other land
        use classes (of which only |ACKER| is selected), only the amount
        of |NBes| exceeding the actual snow holding capacity is passed
        to |WaDa|.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] in (WASSER, FLUSS, SEE):
            sta.waes[k] = 0.
            flu.wada[k] = flu.nbes[k]
        else:
            sta.waes[k] += flu.nbes[k]
            flu.wada[k] = max(sta.waes[k]-con.pwmax[k]*sta.wats[k], 0.)
            sta.waes[k] -= flu.wada[k]