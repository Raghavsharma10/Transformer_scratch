def calc_cf_sm_v1(self):
    """Calculate capillary flow and update soil moisture.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |FC|
      |CFlux|

    Required fluxes sequence:
      |R|

    Required state sequence:
      |UZ|

    Calculated flux sequence:
      |CF|

    Updated state sequence:
      |SM|

    Basic equations:
      :math:`\\frac{dSM}{dt} = CF` \n
      :math:`CF = CFLUX \\cdot (1 - \\frac{SM}{FC})`

    Examples:

        Initialize six zones of different types.  The field
        capacity of als fields and forests is set to 200mm, the maximum
        capillary flow rate is 4mm/d:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nmbzones(6)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD)
        >>> fc(200.0)
        >>> cflux(4.0)

        Note that the assumed length of the simulation step is only
        a half day.  Hence the maximum capillary flow per simulation
        step is 2 instead of 4:

        >>> cflux
        cflux(4.0)
        >>> cflux.values
        array([ 2.,  2.,  2.,  2.,  2.,  2.])

        For fields and forests, the actual capillary return flow depends
        on the relative soil moisture deficite, if either the upper zone
        layer provides enough water...

        >>> fluxes.r = 0.0
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> states.uz = 20.0
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 1.0, 1.0, 2.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 101.0, 101.0, 2.0, 200.0)

        ...our enough effective precipitation is generated, which can be
        rerouted directly:

        >>> cflux(4.0)
        >>> fluxes.r = 10.0
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> states.uz = 0.0
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 1.0, 1.0, 2.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 101.0, 101.0, 2.0, 200.0)

        If the upper zone layer is empty and no effective precipitation is
        generated, capillary flow is zero:

        >>> cflux(4.0)
        >>> fluxes.r = 0.0
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> states.uz = 0.0
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 100.0, 100.0, 0.0, 200.0)

        Here an example, where both the upper zone layer and effective
        precipitation provide water for the capillary flow, but less then
        the maximum flow rate times the relative soil moisture:

        >>> cflux(4.0)
        >>> fluxes.r = 0.1
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> states.uz = 0.2
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 0.3, 0.3, 0.3, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 100.3, 100.3, 0.3, 200.0)

        Even unrealistic high maximum capillary flow rates do not result
        in overfilled soils:

        >>> cflux(1000.0)
        >>> fluxes.r = 200.0
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> states.uz = 200.0
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 100.0, 100.0, 200.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 200.0, 200.0, 200.0, 200.0)

        For (unrealistic) soils with zero field capacity, capillary flow
        is always zero:

        >>> fc(0.0)
        >>> states.sm = 0.0
        >>> model.calc_cf_sm_v1()
        >>> fluxes.cf
        cf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] in (FIELD, FOREST):
            if con.fc[k] > 0.:
                flu.cf[k] = con.cflux[k]*(1.-sta.sm[k]/con.fc[k])
                flu.cf[k] = min(flu.cf[k], sta.uz+flu.r[k])
                flu.cf[k] = min(flu.cf[k], con.fc[k]-sta.sm[k])
            else:
                flu.cf[k] = 0.
            sta.sm[k] += flu.cf[k]
        else:
            flu.cf[k] = 0.
            sta.sm[k] = 0.