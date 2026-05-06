def calc_qib2_v1(self):
    """Calculate the first inflow component released from the soil.

    Required control parameters:
      |NHRU|
      |Lnk|
      |NFk|
      |DMin|
      |DMax|

    Required derived parameter:
      |WZ|

    Required state sequence:
      |BoWa|

    Calculated flux sequence:
      |QIB2|

    Basic equation:
      :math:`QIB2 = (DMax-DMin) \\cdot
      (\\frac{BoWa-WZ}{NFk-WZ})^\\frac{3}{2}`

    Examples:

        For water and sealed areas, no interflow is calculated (the first
        three HRUs are of type |FLUSS|, |SEE|, and |VERS|, respectively).
        No principal distinction is made between the remaining land use
        classes (arable land |ACKER| has been selected for the last
        five HRUs arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(8)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER, ACKER)
        >>> dmax(10.0)
        >>> dmin(4.0)
        >>> nfk(100.0, 100.0, 100.0, 50.0, 100.0, 100.0, 100.0, 200.0)
        >>> derived.wz(50.0)
        >>> states.bowa = 100.0, 100.0, 100.0, 50.1, 50.0, 75.0, 100.0, 100.0

        Note the time dependence of parameters |DMin| (see the example above)
        and |DMax|:

        >>> dmax
        dmax(10.0)
        >>> dmax.values
        array([ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.])

        The following results show that he calculation of |QIB2| both
        resembles those of |QBB| and |QIB1| in some regards:

        >>> model.calc_qib2_v1()
        >>> fluxes.qib2
        qib2(0.0, 0.0, 0.0, 0.0, 0.0, 1.06066, 3.0, 0.57735)

        In the given example, the maximum rate of total interflow
        generation is 5 mm/12h (parameter |DMax|).  For the seventh zone,
        which contains a saturated soil, the value calculated for the
        second interflow component (|QIB2|) is 3 mm/h.  The "missing"
        value of 2 mm/12h is be calculated by method |calc_qib1_v1|.

        (The fourth zone, which is slightly oversaturated, is only intended
        to demonstrate that zero division due to |NFk| = |WZ| is circumvented.)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if ((con.lnk[k] in (VERS, WASSER, FLUSS, SEE)) or
                (sta.bowa[k] <= der.wz[k]) or (con.nfk[k] <= der.wz[k])):
            flu.qib2[k] = 0.
        else:
            flu.qib2[k] = ((con.dmax[k]-con.dmin[k]) *
                           ((sta.bowa[k]-der.wz[k]) /
                            (con.nfk[k]-der.wz[k]))**1.5)