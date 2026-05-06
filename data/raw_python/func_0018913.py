def calc_qigz1_v1(self):
    """Aggregate the amount of the first interflow component released
    by all HRUs.

    Required control parameters:
      |NHRU|
      |FHRU|

    Required flux sequence:
      |QIB1|

    Calculated state sequence:
      |QIGZ1|

    Basic equation:
       :math:`QIGZ1 = \\Sigma(FHRU \\cdot QIB1)`

    Example:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> nhru(2)
        >>> fhru(0.75, 0.25)
        >>> fluxes.qib1 = 1.0, 5.0
        >>> model.calc_qigz1_v1()
        >>> states.qigz1
        qigz1(2.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    sta.qigz1 = 0.
    for k in range(con.nhru):
        sta.qigz1 += con.fhru[k]*flu.qib1[k]