def calc_tf_ic_v1(self):
    """Calculate throughfall and update the interception storage
    accordingly.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |IcMax|

    Required flux sequences:
      |PC|

    Calculated fluxes sequences:
      |TF|

    Updated state sequence:
      |Ic|

    Basic equation:
      :math:`TF = \\Bigl \\lbrace
      {
      {PC \\ | \\ Ic = IcMax}
      \\atop
      {0 \\ | \\ Ic < IcMax}
      }`

    Examples:

        Initialize six zones of different types.  Assume a
        generall maximum interception capacity of 2 mm. All zones receive
        a 0.5 mm input of precipitation:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(6)
        >>> zonetype(GLACIER, ILAKE, FIELD, FOREST, FIELD, FIELD)
        >>> icmax(2.0)
        >>> fluxes.pc = 0.5
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_tf_ic_v1()

        For glaciers (first zone) and internal lakes (second zone) the
        interception routine does not apply.  Hence, all precipitation is
        routed as throughfall. For fields and forests, the interception
        routine is identical (usually, only larger capacities for forests
        are assumed, due to their higher leaf area index).  Hence, the
        results of the third and the second zone are equal.  The last
        three zones demonstrate, that all precipitation is stored until
        the interception capacity is reached; afterwards, all precepitation
        is routed as throughfall.  Initial storage reduces the effective
        capacity of the respective simulation step:

        >>> states.ic
        ic(0.0, 0.0, 0.5, 0.5, 1.5, 2.0)
        >>> fluxes.tf
        tf(0.5, 0.5, 0.0, 0.0, 0.0, 0.5)

        A zero precipitation example:

        >>> fluxes.pc = 0.0
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_tf_ic_v1()
        >>> states.ic
        ic(0.0, 0.0, 0.0, 0.0, 1.0, 2.0)
        >>> fluxes.tf
        tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        A high precipitation example:

        >>> fluxes.pc = 5.0
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_tf_ic_v1()
        >>> states.ic
        ic(0.0, 0.0, 2.0, 2.0, 2.0, 2.0)
        >>> fluxes.tf
        tf(5.0, 5.0, 3.0, 3.0, 4.0, 5.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] in (FIELD, FOREST):
            flu.tf[k] = max(flu.pc[k]-(con.icmax[k]-sta.ic[k]), 0.)
            sta.ic[k] += flu.pc[k]-flu.tf[k]
        else:
            flu.tf[k] = flu.pc[k]
            sta.ic[k] = 0.