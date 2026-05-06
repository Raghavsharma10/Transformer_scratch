def calc_evb_v1(self):
    """Calculate the actual water release from the snow cover.

    Required control parameters:
      |NHRU|
      |Lnk|
      |NFk|
      |GrasRef_R|

    Required state sequence:
      |BoWa|

    Required flux sequences:
      |EvPo|
      |EvI|

    Calculated flux sequence:
      |EvB|

    Basic equations:
      :math:`temp = exp(-GrasRef_R \\cdot \\frac{BoWa}{NFk})`
      :math:`EvB = (EvPo - EvI) \\cdot
      \\frac{1 - temp}{1 + temp -2 \\cdot exp(-GrasRef_R)}`

    Examples:

        Soil evaporation is calculated neither for water nor for sealed
        areas (see the first three HRUs of type |FLUSS|, |SEE|, and |VERS|).
        All other land use classes are handled in accordance with a
        recommendation of the set of codes described in ATV-DVWK-M 504
        (arable land |ACKER| has been selected for the last four HRUs
        arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(7)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER)
        >>> grasref_r(5.0)
        >>> nfk(100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0)
        >>> fluxes.evpo = 5.0
        >>> fluxes.evi = 3.0
        >>> states.bowa = 50.0, 50.0, 50.0, 0.0, 0.0, 50.0, 100.0
        >>> model.calc_evb_v1()
        >>> fluxes.evb
        evb(0.0, 0.0, 0.0, 0.0, 0.0, 1.717962, 2.0)

        In case usable field capacity (|NFk|) is zero, soil evaporation
        (|EvB|) is generally set to zero (see the forth HRU).  The last
        three HRUs demonstrate the rise in soil evaporation with increasing
        soil moisture, which is lessening in the high soil moisture range.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if (con.lnk[k] in (VERS, WASSER, FLUSS, SEE)) or (con.nfk[k] <= 0.):
            flu.evb[k] = 0.
        else:
            d_temp = modelutils.exp(-con.grasref_r *
                                    sta.bowa[k]/con.nfk[k])
            flu.evb[k] = ((flu.evpo[k]-flu.evi[k]) * (1.-d_temp) /
                          (1.+d_temp-2.*modelutils.exp(-con.grasref_r)))