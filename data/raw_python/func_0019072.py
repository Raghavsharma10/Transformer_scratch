def calc_in_wc_v1(self):
    """Calculate the actual water release from the snow layer due to the
    exceedance of the snow layers capacity for (liquid) water.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |WHC|

    Required state sequence:
      |SP|

    Required flux sequence
      |TF|

    Calculated fluxes sequences:
      |In_|

    Updated state sequence:
      |WC|

    Basic equations:
      :math:`\\frac{dWC}{dt} = -In` \n
      :math:`-In = max(WC - WHC \\cdot SP, 0)`

    Examples:

        Initialize six zones of different types and frozen water
        contents of the snow layer and set the relative water holding
        capacity to 20% of the respective frozen water content:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(6)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD)
        >>> whc(0.2)
        >>> states.sp = 0.0, 10.0, 10.0, 10.0, 5.0, 0.0

        Also set the actual value of stand precipitation to 5 mm/d:

        >>> fluxes.tf = 5.0

        When there is no (liquid) water content in the snow layer, no water
        can be released:

        >>> states.wc = 0.0
        >>> model.calc_in_wc_v1()
        >>> fluxes.in_
        in_(5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> states.wc
        wc(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        When there is a (liquid) water content in the snow layer, the water
        release depends on the frozen water content.  Note the special
        cases of the first zone being an internal lake, for which the snow
        routine does not apply, and of the last zone, which has no ice
        content and thus effectively not really a snow layer:

        >>> states.wc = 5.0
        >>> model.calc_in_wc_v1()
        >>> fluxes.in_
        in_(5.0, 3.0, 3.0, 3.0, 4.0, 5.0)
        >>> states.wc
        wc(0.0, 2.0, 2.0, 2.0, 1.0, 0.0)

        When the relative water holding capacity is assumed to be zero,
        all liquid water is released:

        >>> whc(0.0)
        >>> states.wc = 5.0
        >>> model.calc_in_wc_v1()
        >>> fluxes.in_
        in_(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
        >>> states.wc
        wc(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        Note that for the single lake zone, stand precipitation is
        directly passed to `in_` in all three examples.
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] != ILAKE:
            flu.in_[k] = max(sta.wc[k]-con.whc[k]*sta.sp[k], 0.)
            sta.wc[k] -= flu.in_[k]
        else:
            flu.in_[k] = flu.tf[k]
            sta.wc[k] = 0.