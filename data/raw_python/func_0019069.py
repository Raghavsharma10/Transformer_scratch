def calc_ei_ic_v1(self):
    """Calculate interception evaporation and update the interception
    storage accordingly.

    Required control parameters:
      |NmbZones|
      |ZoneType|

    Required flux sequences:
      |EPC|

    Calculated fluxes sequences:
      |EI|

    Updated state sequence:
      |Ic|

    Basic equation:
      :math:`EI = \\Bigl \\lbrace
      {
      {EPC \\ | \\ Ic > 0}
      \\atop
      {0 \\ | \\ Ic = 0}
      }`

    Examples:

        Initialize six zones of different types.  For all zones
        a (corrected) potential evaporation of 0.5 mm is given:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(6)
        >>> zonetype(GLACIER, ILAKE, FIELD, FOREST, FIELD, FIELD)
        >>> fluxes.epc = 0.5
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_ei_ic_v1()

        For glaciers (first zone) and internal lakes (second zone) the
        interception routine does not apply.  Hence, no interception
        evaporation can occur.  For fields and forests, the interception
        routine is identical (usually, only larger capacities for forests
        are assumed, due to their higher leaf area index).  Hence, the
        results of the third and the second zone are equal.  The last
        three zones demonstrate, that all interception evaporation is equal
        to potential evaporation until the interception storage is empty;
        afterwards, interception evaporation is zero:

        >>> states.ic
        ic(0.0, 0.0, 0.0, 0.0, 0.5, 1.5)
        >>> fluxes.ei
        ei(0.0, 0.0, 0.0, 0.0, 0.5, 0.5)

        A zero evaporation example:

        >>> fluxes.epc = 0.0
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_ei_ic_v1()
        >>> states.ic
        ic(0.0, 0.0, 0.0, 0.0, 1.0, 2.0)
        >>> fluxes.ei
        ei(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        A high evaporation example:

        >>> fluxes.epc = 5.0
        >>> states.ic = 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
        >>> model.calc_ei_ic_v1()
        >>> states.ic
        ic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> fluxes.ei
        ei(0.0, 0.0, 0.0, 0.0, 1.0, 2.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] in (FIELD, FOREST):
            flu.ei[k] = min(flu.epc[k], sta.ic[k])
            sta.ic[k] -= flu.ei[k]
        else:
            flu.ei[k] = 0.
            sta.ic[k] = 0.