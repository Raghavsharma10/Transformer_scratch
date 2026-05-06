def calc_qdgz_v1(self):
    """Aggregate the amount of total direct flow released by all HRUs.

    Required control parameters:
      |Lnk|
      |NHRU|
      |FHRU|

    Required flux sequence:
      |QDB|
      |NKor|
      |EvI|

    Calculated flux sequence:
      |QDGZ|

    Basic equation:
       :math:`QDGZ = \\Sigma(FHRU \\cdot QDB) +
       \\Sigma(FHRU \\cdot (NKor_{FLUSS}-EvI_{FLUSS}))`

    Examples:

        The first example shows that |QDGZ| is the area weighted sum of
        |QDB| from "land type" HRUs like arable land (|ACKER|) and sealed
        surfaces (|VERS|) as well as of |NKor|-|EvI| from water areas of
        type |FLUSS|.  Water areas of type |WASSER| and |SEE| have no
        impact on |QDGZ|:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> nhru(5)
        >>> lnk(ACKER, VERS, WASSER, SEE, FLUSS)
        >>> fhru(0.1, 0.2, 0.1, 0.2, 0.4)
        >>> fluxes.qdb = 2., 4.0, 300.0, 300.0, 300.0
        >>> fluxes.nkor = 200.0, 200.0, 200.0, 200.0, 20.0
        >>> fluxes.evi = 100.0, 100.0, 100.0, 100.0, 10.0
        >>> model.calc_qdgz_v1()
        >>> fluxes.qdgz
        qdgz(5.0)

        The second example shows that large evaporation values above a
        HRU of type |FLUSS| can result in negative values of |QDGZ|:

        >>> fluxes.evi[4] = 30
        >>> model.calc_qdgz_v1()
        >>> fluxes.qdgz
        qdgz(-3.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.qdgz = 0.
    for k in range(con.nhru):
        if con.lnk[k] == FLUSS:
            flu.qdgz += con.fhru[k]*(flu.nkor[k]-flu.evi[k])
        elif con.lnk[k] not in (WASSER, SEE):
            flu.qdgz += con.fhru[k]*flu.qdb[k]