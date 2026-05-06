def calc_r_sm_v1(self):
    """Calculate effective precipitation and update soil moisture.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |FC|
      |Beta|

    Required fluxes sequence:
      |In_|

    Calculated flux sequence:
      |R|

    Updated state sequence:
      |SM|

    Basic equations:
      :math:`\\frac{dSM}{dt} = IN - R` \n
      :math:`R = IN \\cdot \\left(\\frac{SM}{FC}\\right)^{Beta}`


    Examples:

        Initialize six zones of different types.  The field
        capacity of all fields and forests is set to 200mm, the input
        of each zone is 10mm:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(6)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD)
        >>> fc(200.0)
        >>> fluxes.in_ = 10.0

        With a common nonlinearity parameter value of 2, a relative
        soil moisture of 50%  (zones three and four) results in a
        discharge coefficient of 25%. For a soil completely dried
        (zone five) or completely saturated (one six) the discharge
        coefficient does not depend on the nonlinearity parameter and
        is 0% and 100% respectively.  Glaciers and internal lakes also
        always route 100% of their input as effective precipitation:

        >>> beta(2.0)
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> model.calc_r_sm_v1()
        >>> fluxes.r
        r(10.0, 10.0, 2.5, 2.5, 0.0, 10.0)
        >>> states.sm
        sm(0.0, 0.0, 107.5, 107.5, 10.0, 200.0)

        Through decreasing the nonlinearity parameter, the discharge
        coefficient increases.  A parameter value of zero leads to a
        discharge coefficient of 100% for any soil moisture:

        >>> beta(0.0)
        >>> states.sm = 0.0, 0.0, 100.0, 100.0, 0.0, 200.0
        >>> model.calc_r_sm_v1()
        >>> fluxes.r
        r(10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        >>> states.sm
        sm(0.0, 0.0, 100.0, 100.0, 0.0, 200.0)

        With zero field capacity, the discharge coefficient also always
        equates to 100%:

        >>> fc(0.0)
        >>> beta(2.0)
        >>> states.sm = 0.0
        >>> model.calc_r_sm_v1()
        >>> fluxes.r
        r(10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        >>> states.sm
        sm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] in (FIELD, FOREST):
            if con.fc[k] > 0.:
                flu.r[k] = flu.in_[k]*(sta.sm[k]/con.fc[k])**con.beta[k]
                flu.r[k] = max(flu.r[k], sta.sm[k]+flu.in_[k]-con.fc[k])
            else:
                flu.r[k] = flu.in_[k]
            sta.sm[k] += flu.in_[k]-flu.r[k]
        else:
            flu.r[k] = flu.in_[k]
            sta.sm[k] = 0.