def calc_evi_inzp_v1(self):
    """Calculate interception evaporation and update the interception
    storage accordingly.

    Required control parameters:
      |NHRU|
      |Lnk|
      |TRefT|
      |TRefN|

    Required flux sequence:
      |EvPo|

    Calculated flux sequence:
      |EvI|

    Updated state sequence:
      |Inzp|

    Basic equation:
      :math:`EvI = \\Bigl \\lbrace
      {
      {EvPo \\ | \\ Inzp > 0}
      \\atop
      {0 \\ | \\ Inzp = 0}
      }`

    Examples:

        Initialize five HRUs with different combinations of land usage
        and initial interception storage and apply a value of potential
        evaporation of 3 mm on each one:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(5)
        >>> lnk(FLUSS, SEE, ACKER, ACKER, ACKER)
        >>> states.inzp = 2.0, 2.0, 0.0, 2.0, 4.0
        >>> fluxes.evpo = 3.0
        >>> model.calc_evi_inzp_v1()
        >>> states.inzp
        inzp(0.0, 0.0, 0.0, 0.0, 1.0)
        >>> fluxes.evi
        evi(3.0, 3.0, 0.0, 2.0, 3.0)

        For arable land (|ACKER|) and most other land types, interception
        evaporation (|EvI|) is identical with potential evapotranspiration
        (|EvPo|), as long as it is met by available intercepted water
        ([Inzp|).  Only water areas (|FLUSS| and |SEE|),  |EvI| is
        generally equal to |EvPo| (but this might be corrected by a method
        called after |calc_evi_inzp_v1| has been applied) and [Inzp| is
        set to zero.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] in (WASSER, FLUSS, SEE):
            flu.evi[k] = flu.evpo[k]
            sta.inzp[k] = 0.
        else:
            flu.evi[k] = min(flu.evpo[k], sta.inzp[k])
            sta.inzp[k] -= flu.evi[k]