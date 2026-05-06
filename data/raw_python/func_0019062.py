def calc_tmean_v1(self):
    """Calculate the areal mean temperature of the subbasin.

    Required derived parameter:
      |RelZoneArea|

    Required flux sequence:
      |TC|

    Calculated flux sequences:
      |TMean|

    Examples:

        Prepare two zones, the first one being twice as large
        as the second one:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(2)
        >>> derived.relzonearea(2.0/3.0, 1.0/3.0)

        With temperature values of 5°C and 8°C  of the respective zones,
        the mean temperature is 6°C:

        >>> fluxes.tc = 5.0, 8.0
        >>> model.calc_tmean_v1()
        >>> fluxes.tmean
        tmean(6.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.tmean = 0.
    for k in range(con.nmbzones):
        flu.tmean += der.relzonearea[k]*flu.tc[k]