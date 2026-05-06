def calc_qdb_v1(self):
    """Calculate direct runoff released from the soil.

    Required control parameters:
      |NHRU|
      |Lnk|
      |NFk|
      |BSf|

    Required state sequence:
      |BoWa|

    Required flux sequence:
      |WaDa|

    Calculated flux sequence:
      |QDB|

    Basic equations:
      :math:`QDB = \\Bigl \\lbrace
      {
      {max(Exz, 0) \\ | \\ SfA \\leq 0}
      \\atop
      {max(Exz + NFk \\cdot SfA^{BSf+1}, 0) \\ | \\ SfA > 0}
      }`
      :math:`SFA = (1 - \\frac{BoWa}{NFk})^\\frac{1}{BSf+1} -
      \\frac{WaDa}{(BSf+1) \\cdot NFk}`
      :math:`Exz = (BoWa + WaDa) - NFk`

    Examples:

        For water areas (|FLUSS| and |SEE|), sealed areas (|VERS|), and
        areas without any soil storage capacity, all water is completely
        routed as direct runoff |QDB| (see the first four HRUs).  No
        principal distinction is made between the remaining land use
        classes (arable land |ACKER| has been selected for the last five
        HRUs arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(9)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER, ACKER, ACKER)
        >>> bsf(0.4)
        >>> nfk(100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0)
        >>> fluxes.wada = 10.0
        >>> states.bowa = (
        ...     100.0, 100.0, 100.0, 0.0, -0.1, 0.0, 50.0, 100.0, 100.1)
        >>> model.calc_qdb_v1()
        >>> fluxes.qdb
        qdb(10.0, 10.0, 10.0, 10.0, 0.142039, 0.144959, 1.993649, 10.0, 10.1)

        With the common |BSf| value of 0.4, the discharge coefficient
        increases more or less exponentially with soil moisture.
        For soil moisture values slightly below zero or above usable
        field capacity, plausible amounts of generated direct runoff
        are ensured.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    aid = self.sequences.aides.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] == WASSER:
            flu.qdb[k] = 0.
        elif ((con.lnk[k] in (VERS, FLUSS, SEE)) or
              (con.nfk[k] <= 0.)):
            flu.qdb[k] = flu.wada[k]
        else:
            if sta.bowa[k] < con.nfk[k]:
                aid.sfa[k] = (
                    (1.-sta.bowa[k]/con.nfk[k])**(1./(con.bsf[k]+1.)) -
                    (flu.wada[k]/((con.bsf[k]+1.)*con.nfk[k])))
            else:
                aid.sfa[k] = 0.
            aid.exz[k] = sta.bowa[k]+flu.wada[k]-con.nfk[k]
            flu.qdb[k] = aid.exz[k]
            if aid.sfa[k] > 0.:
                flu.qdb[k] += aid.sfa[k]**(con.bsf[k]+1.)*con.nfk[k]
            flu.qdb[k] = max(flu.qdb[k], 0.)