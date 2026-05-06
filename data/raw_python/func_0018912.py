def calc_qbgz_v1(self):
    """Aggregate the amount of base flow released by all "soil type" HRUs
    and the "net precipitation" above water areas of type |SEE|.

    Water areas of type |SEE| are assumed to be directly connected with
    groundwater, but not with the stream network.  This is modelled by
    adding their (positive or negative) "net input" (|NKor|-|EvI|) to the
    "percolation output" of the soil containing HRUs.

    Required control parameters:
      |Lnk|
      |NHRU|
      |FHRU|

    Required flux sequences:
      |QBB|
      |NKor|
      |EvI|

    Calculated state sequence:
      |QBGZ|

    Basic equation:
       :math:`QBGZ = \\Sigma(FHRU \\cdot QBB) +
       \\Sigma(FHRU \\cdot (NKor_{SEE}-EvI_{SEE}))`

    Examples:

        The first example shows that |QBGZ| is the area weighted sum of
        |QBB| from "soil type" HRUs like arable land (|ACKER|) and of
        |NKor|-|EvI| from water areas of type |SEE|.  All other water
        areas (|WASSER| and |FLUSS|) and also sealed surfaces (|VERS|)
        have no impact on |QBGZ|:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> nhru(6)
        >>> lnk(ACKER, ACKER, VERS, WASSER, FLUSS, SEE)
        >>> fhru(0.1, 0.2, 0.1, 0.1, 0.1, 0.4)
        >>> fluxes.qbb = 2., 4.0, 300.0, 300.0, 300.0, 300.0
        >>> fluxes.nkor = 200.0, 200.0, 200.0, 200.0, 200.0, 20.0
        >>> fluxes.evi = 100.0, 100.0, 100.0, 100.0, 100.0, 10.0
        >>> model.calc_qbgz_v1()
        >>> states.qbgz
        qbgz(5.0)

        The second example shows that large evaporation values above a
        HRU of type |SEE| can result in negative values of |QBGZ|:

        >>> fluxes.evi[5] = 30
        >>> model.calc_qbgz_v1()
        >>> states.qbgz
        qbgz(-3.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    sta.qbgz = 0.
    for k in range(con.nhru):
        if con.lnk[k] == SEE:
            sta.qbgz += con.fhru[k]*(flu.nkor[k]-flu.evi[k])
        elif con.lnk[k] not in (WASSER, FLUSS, VERS):
            sta.qbgz += con.fhru[k]*flu.qbb[k]