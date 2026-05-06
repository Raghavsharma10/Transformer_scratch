def calc_et0_v1(self):
    """Calculate reference evapotranspiration after Turc-Wendling.

    Required control parameters:
      |NHRU|
      |KE|
      |KF|
      |HNN|

    Required input sequence:
      |Glob|

    Required flux sequence:
      |TKor|

    Calculated flux sequence:
      |ET0|

    Basic equation:
      :math:`ET0 = KE \\cdot
      \\frac{(8.64 \\cdot Glob+93 \\cdot KF) \\cdot (TKor+22)}
      {165 \\cdot (TKor+123) \\cdot (1 + 0.00019 \\cdot min(HNN, 600))}`

    Example:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(3)
        >>> ke(1.1)
        >>> kf(0.6)
        >>> hnn(200.0, 600.0, 1000.0)
        >>> inputs.glob = 200.0
        >>> fluxes.tkor = 15.0
        >>> model.calc_et0_v1()
        >>> fluxes.et0
        et0(3.07171, 2.86215, 2.86215)
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        flu.et0[k] = (con.ke[k]*(((8.64*inp.glob+93.*con.kf[k]) *
                                  (flu.tkor[k]+22.)) /
                                 (165.*(flu.tkor[k]+123.) *
                                  (1.+0.00019*min(con.hnn[k], 600.)))))