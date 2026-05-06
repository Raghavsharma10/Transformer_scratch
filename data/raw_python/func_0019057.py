def calc_hmin_qmin_hmax_qmax_v1(self):
    """Determine an starting interval for iteration methods as the one
    implemented in method |calc_h_v1|.

    The resulting interval is determined in a manner, that on the
    one hand :math:`Qmin \\leq QRef \\leq Qmax` is fulfilled and on the
    other hand the results of method |calc_qg_v1| are continuous
    for :math:`Hmin \\leq H \\leq Hmax`.

    Required control parameter:
      |HM|

    Required derived parameters:
      |HV|
      |lstream_derived.QM|
      |lstream_derived.QV|

    Required flux sequence:
      |QRef|

    Calculated aide sequences:
      |HMin|
      |HMax|
      |QMin|
      |QMax|

    Besides the mentioned required parameters and sequences, those of the
    actual method for calculating the discharge of the total cross section
    might be required.  This is the case whenever water flows on both outer
    embankments.  In such occasions no previously determined upper boundary
    values are available and method |calc_hmin_qmin_hmax_qmax_v1| needs
    to increase the value of :math:`HMax` successively until the condition
    :math:`QG \\leq QMax` is met.
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    if flu.qref <= der.qm:
        aid.hmin = 0.
        aid.qmin = 0.
        aid.hmax = con.hm
        aid.qmax = der.qm
    elif flu.qref <= min(der.qv[0], der.qv[1]):
        aid.hmin = con.hm
        aid.qmin = der.qm
        aid.hmax = con.hm+min(der.hv[0], der.hv[1])
        aid.qmax = min(der.qv[0], der.qv[1])
    elif flu.qref < max(der.qv[0], der.qv[1]):
        aid.hmin = con.hm+min(der.hv[0], der.hv[1])
        aid.qmin = min(der.qv[0], der.qv[1])
        aid.hmax = con.hm+max(der.hv[0], der.hv[1])
        aid.qmax = max(der.qv[0], der.qv[1])
    else:
        flu.h = con.hm+max(der.hv[0], der.hv[1])
        aid.hmin = flu.h
        aid.qmin = flu.qg
        while True:
            flu.h *= 2.
            self.calc_qg()
            if flu.qg < flu.qref:
                aid.hmin = flu.h
                aid.qmin = flu.qg
            else:
                aid.hmax = flu.h
                aid.qmax = flu.qg
                break