def calc_epc_v1(self):
    """Apply the evaporation correction factors and adjust evaporation
    to the altitude of the individual zones.

    Calculate the areal mean of (uncorrected) potential evaporation
    for the subbasin, adjust it to the individual zones in accordance
    with their heights and perform some corrections, among which one
    depends on the actual precipitation.

    Required control parameters:
      |NmbZones|
      |ECorr|
      |ECAlt|
      |ZoneZ|
      |ZRelE|
      |EPF|

    Required flux sequences:
      |EP|
      |PC|

    Calculated flux sequences:
      |EPC|

    Basic equation:
      :math:`EPC = EP \\cdot ECorr
      \\cdot (1+ECAlt \\cdot (ZoneZ-ZRelE))
      \\cdot exp(-EPF \\cdot PC)`


    Examples:

        Four zones are at an elevation of 200 m.  A (uncorrected)
        potential evaporation value of 2 mm and a (corrected) precipitation
        value of 5 mm have been determined for each zone beforehand:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nmbzones(4)
        >>> zrele(2.0)
        >>> zonez(3.0)
        >>> fluxes.ep = 2.0
        >>> fluxes.pc = 5.0

        The first three zones  illustrate the individual evaporation
        corrections due to the general evaporation correction factor
        (|ECorr|, first zone), the altitude correction factor (|ECAlt|,
        second zone), the precipitation related correction factor
        (|EPF|, third zone).  The fourth zone illustrates the interaction
        between all corrections:

        >>> ecorr(1.3, 1.0, 1.0, 1.3)
        >>> ecalt(0.0, 0.1, 0.0, 0.1)
        >>> epf(0.0, 0.0, -numpy.log(0.7)/10.0, -numpy.log(0.7)/10.0)
        >>> model.calc_epc_v1()
        >>> fluxes.epc
        epc(2.6, 1.8, 1.4, 1.638)

        To prevent from calculating negative evaporation values when too
        large values for parameter |ECAlt| are set, a truncation is performed:

        >>> ecalt(2.0)
        >>> model.calc_epc_v1()
        >>> fluxes.epc
        epc(0.0, 0.0, 0.0, 0.0)

    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        flu.epc[k] = (flu.ep[k]*con.ecorr[k] *
                      (1. - con.ecalt[k]*(con.zonez[k]-con.zrele)))
        if flu.epc[k] <= 0.:
            flu.epc[k] = 0.
        else:
            flu.epc[k] *= modelutils.exp(-con.epf[k]*flu.pc[k])