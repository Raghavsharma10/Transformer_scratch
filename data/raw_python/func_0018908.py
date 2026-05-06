def calc_qib1_v1(self):
    """Calculate the first inflow component released from the soil.

    Required control parameters:
      |NHRU|
      |Lnk|
      |NFk|
      |DMin|

    Required derived parameter:
      |WB|

    Required state sequence:
      |BoWa|

    Calculated flux sequence:
      |QIB1|

    Basic equation:
      :math:`QIB1 = DMin \\cdot \\frac{BoWa}{NFk}`

    Examples:

        For water and sealed areas, no interflow is calculated (the first
        three HRUs are of type |FLUSS|, |SEE|, and |VERS|, respectively).
        No principal distinction is made between the remaining land use
        classes (arable land |ACKER| has been selected for the last five
        HRUs arbitrarily):

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(8)
        >>> lnk(FLUSS, SEE, VERS, ACKER, ACKER, ACKER, ACKER, ACKER)
        >>> dmax(10.0)
        >>> dmin(4.0)
        >>> nfk(101.0, 101.0, 101.0, 0.0, 101.0, 101.0, 101.0, 202.0)
        >>> derived.wb(10.0)
        >>> states.bowa = 10.1, 10.1, 10.1, 0.0, 0.0, 10.0, 10.1, 10.1

        Note the time dependence of parameter |DMin|:

        >>> dmin
        dmin(4.0)
        >>> dmin.values
        array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])

        Compared to the calculation of |QBB|, the following results show
        some relevant differences:

        >>> model.calc_qib1_v1()
        >>> fluxes.qib1
        qib1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1)

        Firstly, as demonstrated with the help of the seventh and the
        eight HRU, the generation of the first interflow component |QIB1|
        depends on relative soil moisture.  Secondly, as demonstrated with
        the help the sixth and seventh HRU, it starts abruptly whenever
        the slightest exceedance of the threshold  parameter |WB| occurs.
        Such sharp discontinuouties are a potential source of trouble.
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nhru):
        if ((con.lnk[k] in (VERS, WASSER, FLUSS, SEE)) or
                (sta.bowa[k] <= der.wb[k])):
            flu.qib1[k] = 0.
        else:
            flu.qib1[k] = con.dmin[k]*(sta.bowa[k]/con.nfk[k])