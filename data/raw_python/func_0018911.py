def calc_bowa_v1(self):
    """Update soil moisture and correct fluxes if necessary.

    Required control parameters:
      |NHRU|
      |Lnk|

    Required flux sequence:
      |WaDa|

    Updated state sequence:
      |BoWa|

    Required (and eventually corrected) flux sequences:
      |EvB|
      |QBB|
      |QIB1|
      |QIB2|
      |QDB|

    Basic equations:
       :math:`\\frac{dBoWa}{dt} = WaDa - EvB - QBB - QIB1 - QIB2 - QDB`
       :math:`BoWa \\geq 0`

    Examples:

        For water areas (|FLUSS| and |SEE|) and sealed areas (|VERS|),
        soil moisture |BoWa| is simply set to zero and no flux correction
        are performed (see the first three HRUs).  No principal distinction
        is made between the remaining land use classes (arable land |ACKER|
        has been selected for the last four HRUs arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(7)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER)
        >>> states.bowa = 2.0
        >>> fluxes.wada = 1.0
        >>> fluxes.evb = 1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.3
        >>> fluxes.qbb = 1.0, 1.0, 1.0, 0.0, 0.2, 0.4, 0.6
        >>> fluxes.qib1 = 1.0, 1.0, 1.0, 0.0, 0.3, 0.6, 0.9
        >>> fluxes.qib2 = 1.0, 1.0, 1.0, 0.0, 0.4, 0.8, 1.2
        >>> fluxes.qdb = 1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 1.5
        >>> model.calc_bowa_v1()
        >>> states.bowa
        bowa(0.0, 0.0, 0.0, 3.0, 1.5, 0.0, 0.0)
        >>> fluxes.evb
        evb(1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.2)
        >>> fluxes.qbb
        qbb(1.0, 1.0, 1.0, 0.0, 0.2, 0.4, 0.4)
        >>> fluxes.qib1
        qib1(1.0, 1.0, 1.0, 0.0, 0.3, 0.6, 0.6)
        >>> fluxes.qib2
        qib2(1.0, 1.0, 1.0, 0.0, 0.4, 0.8, 0.8)
        >>> fluxes.qdb
        qdb(1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 1.0)

        For the seventh HRU, the original total loss terms would result in a
        negative soil moisture value.  Hence it is reduced to the total loss
        term of the sixt HRU, which results exactly in a complete emptying
        of the soil storage.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    aid = self.sequences.aides.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] in (VERS, WASSER, FLUSS, SEE):
            sta.bowa[k] = 0.
        else:
            aid.bvl[k] = (
                flu.evb[k]+flu.qbb[k]+flu.qib1[k]+flu.qib2[k]+flu.qdb[k])
            aid.mvl[k] = sta.bowa[k]+flu.wada[k]
            if aid.bvl[k] > aid.mvl[k]:
                aid.rvl[k] = aid.mvl[k]/aid.bvl[k]
                flu.evb[k] *= aid.rvl[k]
                flu.qbb[k] *= aid.rvl[k]
                flu.qib1[k] *= aid.rvl[k]
                flu.qib2[k] *= aid.rvl[k]
                flu.qdb[k] *= aid.rvl[k]
                sta.bowa[k] = 0.
            else:
                sta.bowa[k] = aid.mvl[k]-aid.bvl[k]