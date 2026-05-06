def calc_qbb_v1(self):
    """Calculate the amount of base flow released from the soil.

    Required control parameters:
      |NHRU|
      |Lnk|
      |Beta|
      |FBeta|

    Required derived parameter:
      |WB|
      |WZ|

    Required state sequence:
      |BoWa|

    Calculated flux sequence:
      |QBB|

    Basic equations:
      :math:`Beta_{eff} = \\Bigl \\lbrace
      {
      {Beta \\ | \\ BoWa \\leq WZ}
      \\atop
      {Beta \\cdot (1+(FBeta-1)\\cdot\\frac{BoWa-WZ}{NFk-WZ}) \\|\\ BoWa > WZ}
      }`

      :math:`QBB = \\Bigl \\lbrace
      {
      {0 \\ | \\ BoWa \\leq WB}
      \\atop
      {Beta_{eff}  \\cdot (BoWa - WB) \\|\\ BoWa > WB}
      }`

    Examples:

        For water and sealed areas, no base flow is calculated (see the
        first three HRUs of type |VERS|, |FLUSS|, and |SEE|).  No principal
        distinction is made between the remaining land use classes (arable
        land |ACKER| has been selected for the last five HRUs arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(8)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER, ACKER)
        >>> beta(0.04)
        >>> fbeta(2.0)
        >>> nfk(100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 200.0)
        >>> derived.wb(10.0)
        >>> derived.wz(70.0)

        Note the time dependence of parameter |Beta|:

        >>> beta
        beta(0.04)
        >>> beta.values
        array([ 0.02,  0.02,  0.02,  0.02,  0.02,  0.02,  0.02,  0.02])

        In the first example, the actual soil water content |BoWa| is set
        to low values.  For values below the threshold |WB|, not percolation
        occurs.  Above |WB| (but below |WZ|), |QBB| increases linearly by
        an amount defined by parameter |Beta|:

        >>> states.bowa = 20.0, 20.0, 20.0, 0.0, 0.0, 10.0, 20.0, 20.0
        >>> model.calc_qbb_v1()
        >>> fluxes.qbb
        qbb(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2)

        Note that for the last two HRUs the same amount of
        base flow generation is determined, in spite of the fact
        that both exhibit different relative soil moistures.  It is
        common to modify this "pure absolute dependency" to a "mixed
        absolute/relative dependency" through defining the values of
        parameter |WB| indirectly via parameter |RelWB|.

        In the second example, the actual soil water content |BoWa| is set
        to high values.  For values below threshold |WZ|, the discussion above
        remains valid.  For values above |WZ|, percolation shows a nonlinear
        behaviour when factor |FBeta| is set to values larger than one:

        >>> nfk(0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 200.0)
        >>> states.bowa = 0.0, 0.0, 0.0, 60.0, 70.0, 80.0, 100.0, 200.0
        >>> model.calc_qbb_v1()
        >>> fluxes.qbb
        qbb(0.0, 0.0, 0.0, 1.0, 1.2, 1.866667, 3.6, 7.6)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if ((con.lnk[k] in (VERS, WASSER, FLUSS, SEE)) or
                (sta.bowa[k] <= der.wb[k]) or (con.nfk[k] <= 0.)):
            flu.qbb[k] = 0.
        elif sta.bowa[k] <= der.wz[k]:
            flu.qbb[k] = con.beta[k]*(sta.bowa[k]-der.wb[k])
        else:
            flu.qbb[k] = (con.beta[k]*(sta.bowa[k]-der.wb[k]) *
                          (1.+(con.fbeta[k]-1.)*((sta.bowa[k]-der.wz[k]) /
                                                 (con.nfk[k]-der.wz[k]))))