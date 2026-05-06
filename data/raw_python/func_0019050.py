def calc_qm_v1(self):
    """Calculate the discharge of the main channel after Manning-Strickler.

    Required control parameters:
      |EKM|
      |SKM|
      |Gef|

    Required flux sequence:
      |AM|
      |UM|

    Calculated flux sequence:
      |lstream_fluxes.QM|

    Examples:

        For appropriate strictly positive values:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> ekm(2.0)
        >>> skm(50.0)
        >>> gef(0.01)
        >>> fluxes.am = 3.0
        >>> fluxes.um = 7.0
        >>> model.calc_qm_v1()
        >>> fluxes.qm
        qm(17.053102)

        For zero or negative values of the flown through surface or
        the wetted perimeter:

        >>> fluxes.am = -1.0
        >>> fluxes.um = 7.0
        >>> model.calc_qm_v1()
        >>> fluxes.qm
        qm(0.0)

        >>> fluxes.am = 3.0
        >>> fluxes.um = 0.0
        >>> model.calc_qm_v1()
        >>> fluxes.qm
        qm(0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    if (flu.am > 0.) and (flu.um > 0.):
        flu.qm = con.ekm*con.skm*flu.am**(5./3.)/flu.um**(2./3.)*con.gef**.5
    else:
        flu.qm = 0.