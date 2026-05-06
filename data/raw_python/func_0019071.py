def calc_refr_sp_wc_v1(self):
    """Calculate refreezing of the water content within the snow layer and
    update both the snow layers ice and the water content.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |CFMax|
      |CFR|

    Required derived parameter:
      |TTM|

    Required flux sequences:
      |TC|

    Calculated fluxes sequences:
      |Refr|

    Required state sequence:
      |WC|

    Updated state sequence:
      |SP|

    Basic equations:
      :math:`\\frac{dSP}{dt} =  + Refr` \n
      :math:`\\frac{dWC}{dt} =  - Refr` \n
      :math:`Refr = min(cfr \\cdot cfmax \\cdot (TTM-TC), WC)`

    Examples:

        Six zones are initialized with the same threshold
        temperature, degree day factor and refreezing coefficient, but
        with different zone types and initial states:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nmbzones(6)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD)
        >>> cfmax(4.0)
        >>> cfr(0.1)
        >>> derived.ttm = 2.0
        >>> states.sp = 2.0
        >>> states.wc = 0.0, 1.0, 1.0, 1.0, 0.5, 0.0

        Note that the assumed length of the simulation step is only
        a half day.  Hence the effective value of the degree day
        factor is not 4 but 2:

        >>> cfmax
        cfmax(4.0)
        >>> cfmax.values
        array([ 2.,  2.,  2.,  2.,  2.,  2.])

        When the actual temperature is equal to the threshold
        temperature for melting and refreezing, neither no refreezing
        occurs and the states remain unchanged:

        >>> fluxes.tc = 2.0
        >>> model.calc_refr_sp_wc_v1()
        >>> fluxes.refr
        refr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sp
        sp(0.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        >>> states.wc
        wc(0.0, 1.0, 1.0, 1.0, 0.5, 0.0)

        The same holds true for an actual temperature higher than the
        threshold temperature:

        >>> states.sp = 2.0
        >>> states.wc = 0.0, 1.0, 1.0, 1.0, 0.5, 0.0
        >>> fluxes.tc = 2.0
        >>> model.calc_refr_sp_wc_v1()
        >>> fluxes.refr
        refr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sp
        sp(0.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        >>> states.wc
        wc(0.0, 1.0, 1.0, 1.0, 0.5, 0.0)

        With an actual temperature 3°C above the threshold temperature,
        only melting can occur. Actual melting is consistent with
        potential melting, except for the first zone, which is an
        internal lake, and the last two zones, for which potential
        melting exceeds the available frozen water content of the
        snow layer:

        >>> states.sp = 2.0
        >>> states.wc = 0.0, 1.0, 1.0, 1.0, 0.5, 0.0
        >>> fluxes.tc = 5.0
        >>> model.calc_refr_sp_wc_v1()
        >>> fluxes.refr
        refr(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sp
        sp(0.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        >>> states.wc
        wc(0.0, 1.0, 1.0, 1.0, 0.5, 0.0)

        With an actual temperature 3°C below the threshold temperature,
        refreezing can occur. Actual refreezing is consistent with
        potential refreezing, except for the first zone, which is an
        internal lake, and the last two zones, for which potential
        refreezing exceeds the available liquid water content of the
        snow layer:

        >>> states.sp = 2.0
        >>> states.wc = 0.0, 1.0, 1.0, 1.0, 0.5, 0.0
        >>> fluxes.tc = -1.0
        >>> model.calc_refr_sp_wc_v1()
        >>> fluxes.refr
        refr(0.0, 0.6, 0.6, 0.6, 0.5, 0.0)
        >>> states.sp
        sp(0.0, 2.6, 2.6, 2.6, 2.5, 2.0)
        >>> states.wc
        wc(0.0, 0.4, 0.4, 0.4, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] != ILAKE:
            if flu.tc[k] < der.ttm[k]:
                flu.refr[k] = min(con.cfr[k]*con.cfmax[k] *
                                  (der.ttm[k]-flu.tc[k]), sta.wc[k])
                sta.sp[k] += flu.refr[k]
                sta.wc[k] -= flu.refr[k]
            else:
                flu.refr[k] = 0.

        else:
            flu.refr[k] = 0.
            sta.wc[k] = 0.
            sta.sp[k] = 0.