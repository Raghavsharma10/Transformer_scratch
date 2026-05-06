def calc_schm_wats_v1(self):
    """Calculate the actual amount of water melting within the snow cover.

    Required control parameters:
      |NHRU|
      |Lnk|

    Required flux sequences:
      |SBes|
      |WGTF|

    Calculated flux sequence:
      |Schm|

    Updated state sequence:
      |WATS|

    Basic equations:
      :math:`\\frac{dWATS}{dt}  = SBes - Schm`
      :math:`Schm = \\Bigl \\lbrace
      {
      {WGTF \\ | \\ WATS > 0}
      \\atop
      {0 \\ | \\ WATS = 0}
      }`

    Examples:

        Initialize two water (|FLUSS| and |SEE|) and four arable land
        (|ACKER|) HRUs.  Assume the same values for the initial amount
        of frozen water (|WATS|) and the frozen part of stand precipitation
        (|SBes|), but different values for potential snowmelt (|WGTF|):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(6)
        >>> lnk(FLUSS, SEE, ACKER, ACKER, ACKER, ACKER)
        >>> states.wats = 2.0
        >>> fluxes.sbes = 1.0
        >>> fluxes.wgtf = 1.0, 1.0, 0.0, 1.0, 3.0, 5.0
        >>> model.calc_schm_wats_v1()
        >>> states.wats
        wats(0.0, 0.0, 3.0, 2.0, 0.0, 0.0)
        >>> fluxes.schm
        schm(0.0, 0.0, 0.0, 1.0, 3.0, 3.0)

        For the water areas, both the frozen amount of water and actual melt
        are set to zero.  For all other land use classes, actual melt
        is either limited by potential melt or the available frozen water,
        which is the sum of initial frozen water and the frozen part
        of stand precipitation.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] in (WASSER, FLUSS, SEE):
            sta.wats[k] = 0.
            flu.schm[k] = 0.
        else:
            sta.wats[k] += flu.sbes[k]
            flu.schm[k] = min(flu.wgtf[k], sta.wats[k])
            sta.wats[k] -= flu.schm[k]