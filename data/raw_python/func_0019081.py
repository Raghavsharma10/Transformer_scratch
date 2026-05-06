def calc_el_lz_v1(self):
    """Calculate lake evaporation.

    Required control parameters:
        |NmbZones|
        |ZoneType|
        |TTIce|

    Required derived parameters:
        |RelZoneArea|

    Required fluxes sequences:
        |TC|
        |EPC|

    Updated state sequence:
        |LZ|

    Basic equations:
        :math:`\\frac{dLZ}{dt} = -EL` \n
        :math:`EL = \\Bigl \\lbrace
        {
        {EPC \\ | \\ TC > TTIce}
        \\atop
        {0 \\ | \\ TC \\leq TTIce}
        }`

    Examples:

        Six zones of the same size are initialized.  The first three
        zones are no internal lakes, they can not exhibit any lake
        evaporation.  Of the last three zones, which are internal lakes,
        only the last one evaporates water.  For zones five and six,
        evaporation is suppressed due to an assumed ice layer, whenever
        the associated theshold temperature is not exceeded:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(6)
        >>> zonetype(FIELD, FOREST, GLACIER, ILAKE, ILAKE, ILAKE)
        >>> ttice(-1.0)
        >>> derived.relzonearea = 1.0/6.0
        >>> fluxes.epc = 0.6
        >>> fluxes.tc = 0.0, 0.0, 0.0, 0.0, -1.0, -2.0
        >>> states.lz = 10.0
        >>> model.calc_el_lz_v1()
        >>> fluxes.el
        el(0.0, 0.0, 0.0, 0.6, 0.0, 0.0)
        >>> states.lz
        lz(9.9)

        Note that internal lakes always contain water.  Hence, the
        HydPy-H-Land model allows for negative values of the lower
        zone storage:

        >>> states.lz = 0.05
        >>> model.calc_el_lz_v1()
        >>> fluxes.el
        el(0.0, 0.0, 0.0, 0.6, 0.0, 0.0)
        >>> states.lz
        lz(-0.05)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if (con.zonetype[k] == ILAKE) and (flu.tc[k] > con.ttice[k]):
            flu.el[k] = flu.epc[k]
            sta.lz -= der.relzonearea[k]*flu.el[k]
        else:
            flu.el[k] = 0.