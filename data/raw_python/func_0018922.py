def calc_q_v1(self):
    """Calculate the final runoff.

    Note that, in case there are water areas, their |NKor| values are
    added and their |EvPo| values are subtracted from the "potential"
    runoff value, if possible.  This hold true for |WASSER| only and is
    due to compatibility with the original LARSIM implementation. Using land
    type |WASSER| can result  in problematic modifications of simulated
    runoff series. It seems advisable to use land type |FLUSS| and/or
    land type |SEE| instead.

    Required control parameters:
      |NHRU|
      |FHRU|
      |Lnk|
      |NegQ|

    Required flux sequence:
      |NKor|

    Updated flux sequence:
      |EvI|

    Required state sequences:
      |QBGA|
      |QIGA1|
      |QIGA2|
      |QDGA1|
      |QDGA2|

    Calculated flux sequence:
      |lland_fluxes.Q|

    Basic equations:
       :math:`Q = QBGA + QIGA1 + QIGA2 + QDGA1 + QDGA2 +
       NKor_{WASSER} - EvI_{WASSER}`
       :math:`Q \\geq 0`

    Examples:

        When there are no water areas in the respective subbasin (we
        choose arable land |ACKER| arbitrarily), the different runoff
        components are simply summed up:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> nhru(3)
        >>> lnk(ACKER, ACKER, ACKER)
        >>> fhru(0.5, 0.2, 0.3)
        >>> negq(False)
        >>> states.qbga = 0.1
        >>> states.qiga1 = 0.3
        >>> states.qiga2 = 0.5
        >>> states.qdga1 = 0.7
        >>> states.qdga2 = 0.9
        >>> fluxes.nkor = 10.0
        >>> fluxes.evi = 4.0, 5.0, 3.0
        >>> model.calc_q_v1()
        >>> fluxes.q
        q(2.5)
        >>> fluxes.evi
        evi(4.0, 5.0, 3.0)

        The defined values of interception evaporation do not show any
        impact on the result of the given example, the predefined values
        for sequence |EvI| remain unchanged.  But when the first HRU is
        assumed to be a water area (|WASSER|), its adjusted precipitaton
        |NKor| value and its interception  evaporation |EvI| value are added
        to and subtracted from |lland_fluxes.Q| respectively:

        >>> control.lnk(WASSER, VERS, NADELW)
        >>> model.calc_q_v1()
        >>> fluxes.q
        q(5.5)
        >>> fluxes.evi
        evi(4.0, 5.0, 3.0)

        Note that only 5 mm are added (instead of the |NKor| value 10 mm)
        and that only 2 mm are substracted (instead of the |EvI| value 4 mm,
        as the first HRU`s area only accounts for 50 % of the subbasin area.

        Setting also the land use class of the second HRU to land type
        |WASSER| and resetting |NKor| to zero would result in overdrying.
        To avoid this, both actual water evaporation values stored in
        sequence |EvI| are reduced by the same factor:

        >>> control.lnk(WASSER, WASSER, NADELW)
        >>> fluxes.nkor = 0.0
        >>> model.calc_q_v1()
        >>> fluxes.q
        q(0.0)
        >>> fluxes.evi
        evi(3.333333, 4.166667, 3.0)

        The handling from water areas of type |FLUSS| and |SEE| differs
        from those of type |WASSER|, as these do receive their net input
        before the runoff concentration routines are applied.  This
        should be more realistic in most cases (especially for type |SEE|
        representing lakes not direct connected to the stream network).
        But it could sometimes result in negative outflow values. This
        is avoided by simply setting |lland_fluxes.Q| to zero and adding
        the truncated negative outflow value to the |EvI| value of all
        HRUs of type |FLUSS| and |SEE|:

        >>> control.lnk(FLUSS, SEE, NADELW)
        >>> states.qbga = -1.0
        >>> states.qdga2 = -1.5
        >>> fluxes.evi = 4.0, 5.0, 3.0
        >>> model.calc_q_v1()
        >>> fluxes.q
        q(0.0)
        >>> fluxes.evi
        evi(2.571429, 3.571429, 3.0)

        This adjustment of |EvI| is only correct regarding the total
        water balance.  Neither spatial nor temporal consistency of the
        resulting |EvI| values are assured.  In the most extreme case,
        even negative |EvI| values might occur.  This seems acceptable,
        as long as the adjustment of |EvI| is rarely triggered.  When in
        doubt about this, check sequences |EvPo| and |EvI| of HRUs of
        types |FLUSS| and |SEE| for possible discrepancies.  Also note
        that there might occur unnecessary corrections of |lland_fluxes.Q|
        in case landtype |WASSER| is combined with either landtype
        |SEE| or |FLUSS|.

        Eventually you might want to avoid correcting |lland_fluxes.Q|.
        This can be achieved by setting parameter |NegQ| to `True`:

        >>> negq(True)
        >>> fluxes.evi = 4.0, 5.0, 3.0
        >>> model.calc_q_v1()
        >>> fluxes.q
        q(-1.0)
        >>> fluxes.evi
        evi(4.0, 5.0, 3.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    aid = self.sequences.aides.fastaccess
    flu.q = sta.qbga+sta.qiga1+sta.qiga2+sta.qdga1+sta.qdga2
    if (not con.negq) and (flu.q < 0.):
        d_area = 0.
        for k in range(con.nhru):
            if con.lnk[k] in (FLUSS, SEE):
                d_area += con.fhru[k]
        if d_area > 0.:
            for k in range(con.nhru):
                if con.lnk[k] in (FLUSS, SEE):
                    flu.evi[k] += flu.q/d_area
        flu.q = 0.
    aid.epw = 0.
    for k in range(con.nhru):
        if con.lnk[k] == WASSER:
            flu.q += con.fhru[k]*flu.nkor[k]
            aid.epw += con.fhru[k]*flu.evi[k]
    if (flu.q > aid.epw) or con.negq:
        flu.q -= aid.epw
    elif aid.epw > 0.:
        for k in range(con.nhru):
            if con.lnk[k] == WASSER:
                flu.evi[k] *= flu.q/aid.epw
        flu.q = 0.