def calc_glmelt_in_v1(self):
    """Calculate melting from glaciers which are actually not covered by
    a snow layer and add it to the water release of the snow module.

    Required control parameters:
      |NmbZones|
      |ZoneType|
      |GMelt|

    Required state sequence:
      |SP|

    Required flux sequence:
      |TC|

    Calculated fluxes sequence:
      |GlMelt|

    Updated flux sequence:
      |In_|

    Basic equation:

      :math:`GlMelt = \\Bigl \\lbrace
      {
      {max(GMelt \\cdot (TC-TTM), 0) \\ | \\ SP = 0}
      \\atop
      {0 \\ | \\ SP > 0}
      }`


    Examples:

        Seven zones are prepared, but glacier melting occurs only
        in the fourth one, as the first three zones are no glaciers, the
        fifth zone is covered by a snow layer and the actual temperature
        of the last two zones is not above the threshold temperature:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nmbzones(7)
        >>> zonetype(FIELD, FOREST, ILAKE, GLACIER, GLACIER, GLACIER, GLACIER)
        >>> gmelt(4.)
        >>> derived.ttm(2.)
        >>> states.sp = 0., 0., 0., 0., .1, 0., 0.
        >>> fluxes.tc = 3., 3., 3., 3., 3., 2., 1.
        >>> fluxes.in_ = 3.
        >>> model.calc_glmelt_in_v1()
        >>> fluxes.glmelt
        glmelt(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
        >>> fluxes.in_
        in_(3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0)

        Note that the assumed length of the simulation step is only
        a half day. Hence the effective value of the degree day factor
        is not 4 but 2:

        >>> gmelt
        gmelt(4.0)
        >>> gmelt.values
        array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.])
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if ((con.zonetype[k] == GLACIER) and
                (sta.sp[k] <= 0.) and (flu.tc[k] > der.ttm[k])):
            flu.glmelt[k] = con.gmelt[k]*(flu.tc[k]-der.ttm[k])
            flu.in_[k] += flu.glmelt[k]
        else:
            flu.glmelt[k] = 0.