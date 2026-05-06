def calc_contriarea_v1(self):
    """Determine the relative size of the contributing area of the whole
    subbasin.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |RespArea|
      |FC|
      |Beta|

    Required derived parameter:
    |RelSoilArea|

    Required state sequence:
      |SM|

    Calculated fluxes sequences:
      |ContriArea|

    Basic equation:
      :math:`ContriArea = \\left( \\frac{SM}{FC} \\right)^{Beta}`

    Examples:
        Four zones are initialized, but only the first two zones
        of type field and forest are taken into account in the calculation
        of the relative contributing area of the catchment (even, if also
        glaciers contribute to the inflow of the upper zone layer):

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(4)
        >>> zonetype(FIELD, FOREST, GLACIER, ILAKE)
        >>> beta(2.0)
        >>> fc(200.0)
        >>> resparea(True)
        >>> derived.relsoilarea(0.5)
        >>> derived.relsoilzonearea(1.0/3.0, 2.0/3.0, 0.0, 0.0)

        With a relative soil moisture of 100 % in the whole subbasin, the
        contributing area is also estimated as 100 %,...

        >>> states.sm = 200.0
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(1.0)

        ...and relative soil moistures of 0% result in an contributing
        area of 0 %:

        >>> states.sm = 0.0
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(0.0)

        With the given value 2 of the nonlinearity parameter Beta, soil
        moisture of 50 % results in a contributing area estimate of 25%:

        >>> states.sm = 100.0
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(0.25)

        Setting the response area option to False,...

        >>> resparea(False)
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(1.0)

        ... setting the soil area (total area of all field and forest
        zones in the subbasin) to zero...,

        >>> resparea(True)
        >>> derived.relsoilarea(0.0)
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(1.0)

        ...or setting all field capacities to zero...

        >>> derived.relsoilarea(0.5)
        >>> fc(0.0)
        >>> states.sm = 0.0
        >>> model.calc_contriarea_v1()
        >>> fluxes.contriarea
        contriarea(1.0)

        ...leads to contributing area values of 100 %.
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    if con.resparea and (der.relsoilarea > 0.):
        flu.contriarea = 0.
        for k in range(con.nmbzones):
            if con.zonetype[k] in (FIELD, FOREST):
                if con.fc[k] > 0.:
                    flu.contriarea += (der.relsoilzonearea[k] *
                                       (sta.sm[k]/con.fc[k])**con.beta[k])
                else:
                    flu.contriarea += der.relsoilzonearea[k]
    else:
        flu.contriarea = 1.