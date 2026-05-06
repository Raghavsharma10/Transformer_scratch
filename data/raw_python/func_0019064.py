def calc_rfc_sfc_v1(self):
    """Calculate the corrected fractions rainfall/snowfall and total
    precipitation.

    Required control parameters:
      |NmbZones|
      |RfCF|
      |SfCF|

    Calculated flux sequences:
      |RfC|
      |SfC|

    Basic equations:
      :math:`RfC = RfCF \\cdot FracRain` \n
      :math:`SfC = SfCF \\cdot (1 - FracRain)`

    Examples:

        Assume five zones with different temperatures and hence
        different fractions of rainfall and total precipitation:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(5)
        >>> fluxes.fracrain = 0.0, 0.25, 0.5, 0.75, 1.0

        With no rainfall and no snowfall correction (implied by the
        respective factors being one), the corrected fraction related
        to rainfall is identical with the original fraction and the
        corrected fraction related to snowfall behaves opposite:

        >>> rfcf(1.0)
        >>> sfcf(1.0)
        >>> model.calc_rfc_sfc_v1()
        >>> fluxes.rfc
        rfc(0.0, 0.25, 0.5, 0.75, 1.0)
        >>> fluxes.sfc
        sfc(1.0, 0.75, 0.5, 0.25, 0.0)

        With a negative rainfall correction of 20% and a positive
        snowfall correction of 20 % the corrected fractions are:

        >>> rfcf(0.8)
        >>> sfcf(1.2)
        >>> model.calc_rfc_sfc_v1()
        >>> fluxes.rfc
        rfc(0.0, 0.2, 0.4, 0.6, 0.8)
        >>> fluxes.sfc
        sfc(1.2, 0.9, 0.6, 0.3, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        flu.rfc[k] = flu.fracrain[k]*con.rfcf[k]
        flu.sfc[k] = (1.-flu.fracrain[k])*con.sfcf[k]