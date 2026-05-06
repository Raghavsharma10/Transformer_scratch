def calc_wgtf_v1(self):
    """Calculate the potential snowmelt.

    Required control parameters:
      |NHRU|
      |Lnk|
      |GTF|
      |TRefT|
      |TRefN|
      |RSchmelz|
      |CPWasser|

    Required flux sequence:
      |TKor|

    Calculated fluxes sequence:
      |WGTF|

    Basic equation:
      :math:`WGTF = max(GTF \\cdot (TKor - TRefT), 0) +
      max(\\frac{CPWasser}{RSchmelz} \\cdot (TKor - TRefN), 0)`

    Examples:

        Initialize seven HRUs with identical degree-day factors and
        temperature thresholds, but different combinations of land use
        and air temperature:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(7)
        >>> lnk(ACKER, LAUBW, FLUSS, SEE, ACKER, ACKER, ACKER)
        >>> gtf(5.0)
        >>> treft(0.0)
        >>> trefn(1.0)
        >>> fluxes.tkor = 2.0, 2.0, 2.0, 2.0, -1.0, 0.0, 1.0

        Compared to most other LARSIM parameters, the specific heat capacity
        and melt heat capacity of water can be seen as fixed properties:

        >>> cpwasser(4.1868)
        >>> rschmelz(334.0)

        Note that the values of the degree-day factor are only half
        as much as the given value, due to the simulation step size
        being only half as long as the parameter step size:

        >>> gtf
        gtf(5.0)
        >>> gtf.values
        array([ 2.5,  2.5,  2.5,  2.5,  2.5,  2.5,  2.5])

        After performing the calculation, one can see that the potential
        melting rate is identical for the first two HRUs (|ACKER| and
        |LAUBW|).  The land use class results in no difference, except for
        water areas (third and forth HRU, |FLUSS| and |SEE|), where no
        potential melt needs to be calculated.  The last three HRUs (again
        |ACKER|) show the usual behaviour of the degree day method, when the
        actual temperature is below (fourth HRU), equal to (fifth HRU) or
        above (sixths zone) the threshold temperature.  Additionally, the
        first two zones show the influence of the additional energy intake
        due to "warm" precipitation.  Obviously, this additional term is
        quite negligible for common parameterizations, even if lower
        values for the separate threshold temperature |TRefT| would be
        taken into account:

        >>> model.calc_wgtf_v1()
        >>> fluxes.wgtf
        wgtf(5.012535, 5.012535, 0.0, 0.0, 0.0, 0.0, 2.5)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        if con.lnk[k] in (WASSER, FLUSS, SEE):
            flu.wgtf[k] = 0.
        else:
            flu.wgtf[k] = (
                max(con.gtf[k]*(flu.tkor[k]-con.treft[k]), 0) +
                max(con.cpwasser/con.rschmelz*(flu.tkor[k]-con.trefn[k]), 0.))