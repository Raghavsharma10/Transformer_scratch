def calc_ea_sm_v1(self):
    """Calculate soil evaporation and update soil moisture.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |FC|
      |LP|
      |ERed|

    Required fluxes sequences:
      |EPC|
      |EI|

    Required state sequence:
      |SP|

    Calculated flux sequence:
      |EA|

    Updated state sequence:
      |SM|

    Basic equations:
      :math:`\\frac{dSM}{dt} = - EA` \n
      :math:`EA_{temp} = \\biggl \\lbrace
      {
      {EPC \\cdot min\\left(\\frac{SM}{LP \\cdot FC}, 1\\right)
      \\ | \\ SP = 0}
      \\atop
      {0 \\ | \\ SP > 0}
      }` \n
      :math:`EA = EA_{temp} - max(ERED \\cdot (EA_{temp} + EI - EPC), 0)`

    Examples:

        Initialize seven zones of different types.  The field capacity
         of all fields and forests is set to 200mm, potential evaporation
         and interception evaporation are 2mm and 1mm respectively:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(7)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD, FIELD)
        >>> fc(200.0)
        >>> lp(0.0, 0.0, 0.5, 0.5, 0.0, 0.8, 1.0)
        >>> ered(0.0)
        >>> fluxes.epc = 2.0
        >>> fluxes.ei = 1.0
        >>> states.sp = 0.0

        Only fields and forests include soils; for glaciers and zones (the
        first two zones) no soil evaporation is performed.  For fields and
        forests, the underlying calculations are the same. In the following
        example, the relative soil moisture is 50% in all field and forest
        zones.  Hence, differences in soil evaporation are related to the
        different soil evaporation parameter values only:

        >>> states.sm = 100.0
        >>> model.calc_ea_sm_v1()
        >>> fluxes.ea
        ea(0.0, 0.0, 2.0, 2.0, 2.0, 1.25, 1.0)
        >>> states.sm
        sm(0.0, 0.0, 98.0, 98.0, 98.0, 98.75, 99.0)

        In the last example, evaporation values of 2mm have been calculated
        for some zones despite the fact, that these 2mm added to the actual
        interception evaporation of 1mm exceed potential evaporation.  This
        behaviour can be reduced...

        >>> states.sm = 100.0
        >>> ered(0.5)
        >>> model.calc_ea_sm_v1()
        >>> fluxes.ea
        ea(0.0, 0.0, 1.5, 1.5, 1.5, 1.125, 1.0)
        >>> states.sm
        sm(0.0, 0.0, 98.5, 98.5, 98.5, 98.875, 99.0)

        ...or be completely excluded:

        >>> states.sm = 100.0
        >>> ered(1.0)
        >>> model.calc_ea_sm_v1()
        >>> fluxes.ea
        ea(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        >>> states.sm
        sm(0.0, 0.0, 99.0, 99.0, 99.0, 99.0, 99.0)

        Any occurrence of a snow layer suppresses soil evaporation
        completely:

        >>> states.sp = 0.01
        >>> states.sm = 100.0
        >>> model.calc_ea_sm_v1()
        >>> fluxes.ea
        ea(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0)

        For (unrealistic) soils with zero field capacity, soil evaporation
        is always zero:

        >>> fc(0.0)
        >>> states.sm = 0.0
        >>> model.calc_ea_sm_v1()
        >>> fluxes.ea
        ea(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.sm
        sm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] in (FIELD, FOREST):
            if sta.sp[k] <= 0.:
                if (con.lp[k]*con.fc[k]) > 0.:
                    flu.ea[k] = flu.epc[k]*sta.sm[k]/(con.lp[k]*con.fc[k])
                    flu.ea[k] = min(flu.ea[k], flu.epc[k])
                else:
                    flu.ea[k] = flu.epc[k]
                flu.ea[k] -= max(con.ered[k] *
                                 (flu.ea[k]+flu.ei[k]-flu.epc[k]), 0.)
                flu.ea[k] = min(flu.ea[k], sta.sm[k])
            else:
                flu.ea[k] = 0.
            sta.sm[k] -= flu.ea[k]
        else:
            flu.ea[k] = 0.
            sta.sm[k] = 0.