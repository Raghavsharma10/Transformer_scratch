def calc_lz_v1(self):
    """Update the lower zone layer in accordance with percolation from
    upper groundwater to lower groundwater and/or in accordance with
    lake precipitation.

    Required control parameters:
      |NmbZones|
      |ZoneType|

    Required derived parameters:
      |RelLandArea|
      |RelZoneArea|

    Required fluxes sequences:
      |PC|
      |Perc|

    Updated state sequence:
      |LZ|

    Basic equation:
      :math:`\\frac{dLZ}{dt} = Perc + Pc`

    Examples:

        At first, a subbasin with two field zones is assumed (the zones
        could be of type forest or glacier as well).  In such zones,
        precipitation does not fall directly into the lower zone layer,
        hence the given precipitation of 2mm has no impact.  Only
        the actual percolation from the upper zone layer (underneath
        both field zones) is added to the lower zone storage:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(2)
        >>> zonetype(FIELD, FIELD)
        >>> derived.rellandarea = 1.0
        >>> derived.relzonearea = 2.0/3.0, 1.0/3.0
        >>> fluxes.perc = 2.0
        >>> fluxes.pc = 5.0
        >>> states.lz = 10.0
        >>> model.calc_lz_v1()
        >>> states.lz
        lz(12.0)

        If the second zone is an internal lake, its precipitation falls
        on the lower zone layer directly.  Note that only 5/3mm
        precipitation are added, due to the relative size of the
        internal lake within the subbasin. Percolation from the upper
        zone layer increases the lower zone storage only by two thirds
        of its original value, due to the larger spatial extend of
        the lower zone layer:

        >>> zonetype(FIELD, ILAKE)
        >>> derived.rellandarea = 2.0/3.0
        >>> derived.relzonearea = 2.0/3.0, 1.0/3.0
        >>> states.lz = 10.0
        >>> model.calc_lz_v1()
        >>> states.lz
        lz(13.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    sta.lz += der.rellandarea*flu.perc
    for k in range(con.nmbzones):
        if con.zonetype[k] == ILAKE:
            sta.lz += der.relzonearea[k]*flu.pc[k]