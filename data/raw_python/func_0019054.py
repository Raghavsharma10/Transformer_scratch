def calc_qvr_v1(self):
    """Calculate the discharge of both outer embankments after
    Manning-Strickler.

    Required control parameters:
      |EKV|
      |SKV|
      |Gef|

    Required flux sequence:
      |AVR|
      |UVR|

    Calculated flux sequence:
      |QVR|

    Examples:

        For appropriate strictly positive values:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> ekv(2.0)
        >>> skv(50.0)
        >>> gef(0.01)
        >>> fluxes.avr = 3.0
        >>> fluxes.uvr = 7.0
        >>> model.calc_qvr_v1()
        >>> fluxes.qvr
        qvr(17.053102, 17.053102)

        For zero or negative values of the flown through surface or
        the wetted perimeter:

        >>> fluxes.avr = -1.0, 3.0
        >>> fluxes.uvr = 7.0, 0.0
        >>> model.calc_qvr_v1()
        >>> fluxes.qvr
        qvr(0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for i in range(2):
        if (flu.avr[i] > 0.) and (flu.uvr[i] > 0.):
            flu.qvr[i] = (con.ekv[i]*con.skv[i] *
                          flu.avr[i]**(5./3.)/flu.uvr[i]**(2./3.)*con.gef**.5)
        else:
            flu.qvr[i] = 0.