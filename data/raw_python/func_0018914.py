def calc_qigz2_v1(self):
    """Aggregate the amount of the second interflow component released
    by all HRUs.

    Required control parameters:
      |NHRU|
      |FHRU|

    Required flux sequence:
      |QIB2|

    Calculated state sequence:
      |QIGZ2|

    Basic equation:
       :math:`QIGZ2 = \\Sigma(FHRU \\cdot QIB2)`

    Example:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> nhru(2)
        >>> fhru(0.75, 0.25)
        >>> fluxes.qib2 = 1.0, 5.0
        >>> model.calc_qigz2_v1()
        >>> states.qigz2
        qigz2(2.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    sta.qigz2 = 0.
    for k in range(con.nhru):
        sta.qigz2 += con.fhru[k]*flu.qib2[k]