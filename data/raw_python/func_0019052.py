def calc_qv_v1(self):
    """Calculate the discharge of both forelands after Manning-Strickler.

    Required control parameters:
      |EKV|
      |SKV|
      |Gef|

    Required flux sequence:
      |AV|
      |UV|

    Calculated flux sequence:
      |lstream_fluxes.QV|

    Examples:

        For appropriate strictly positive values:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> ekv(2.0)
        >>> skv(50.0)
        >>> gef(0.01)
        >>> fluxes.av = 3.0
        >>> fluxes.uv = 7.0
        >>> model.calc_qv_v1()
        >>> fluxes.qv
        qv(17.053102, 17.053102)

        For zero or negative values of the flown through surface or
        the wetted perimeter:

        >>> fluxes.av = -1.0, 3.0
        >>> fluxes.uv = 7.0, 0.0
        >>> model.calc_qv_v1()
        >>> fluxes.qv
        qv(0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for i in range(2):
        if (flu.av[i] > 0.) and (flu.uv[i] > 0.):
            flu.qv[i] = (con.ekv[i]*con.skv[i] *
                         flu.av[i]**(5./3.)/flu.uv[i]**(2./3.)*con.gef**.5)
        else:
            flu.qv[i] = 0.