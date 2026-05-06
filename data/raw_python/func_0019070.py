def calc_sp_wc_v1(self):
    """Add throughfall to the snow layer.

    Required control parameters:
      |NmbZones|
      |ZoneType|

    Required flux sequences:
      |TF|
      |RfC|
      |SfC|

    Updated state sequences:
      |WC|
      |SP|

    Basic equations:
      :math:`\\frac{dSP}{dt} = TF \\cdot \\frac{SfC}{SfC+RfC}` \n
      :math:`\\frac{dWC}{dt} = TF \\cdot \\frac{RfC}{SfC+RfC}`

    Exemples:

        Consider the following setting, in which eight zones of
        different type receive a throughfall of 10mm:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(8)
        >>> zonetype(ILAKE, GLACIER, FIELD, FOREST, FIELD, FIELD, FIELD, FIELD)
        >>> fluxes.tf = 10.0
        >>> fluxes.sfc = 0.5, 0.5, 0.5, 0.5, 0.2, 0.8, 1.0, 4.0
        >>> fluxes.rfc = 0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 4.0, 1.0
        >>> states.sp = 0.0
        >>> states.wc = 0.0
        >>> model.calc_sp_wc_v1()
        >>> states.sp
        sp(0.0, 5.0, 5.0, 5.0, 2.0, 8.0, 2.0, 8.0)
        >>> states.wc
        wc(0.0, 5.0, 5.0, 5.0, 8.0, 2.0, 8.0, 2.0)

        The snow routine does not apply for internal lakes, which is why
        both  the ice storage and the water storage of the first zone
        remain unchanged.  The snow routine is identical for glaciers,
        fields and forests in the current context, which is why the
        results of the second, third, and fourth zone are equal.  The
        last four zones illustrate that the corrected snowfall fraction
        as well as the corrected rainfall fraction are applied in a
        relative manner, as the total amount of water yield has been
        corrected in the interception module already.

        When both factors are zero, the neither the water nor the ice
        content of the snow layer changes:

        >>> fluxes.sfc = 0.0
        >>> fluxes.rfc = 0.0
        >>> states.sp = 2.0
        >>> states.wc = 0.0
        >>> model.calc_sp_wc_v1()
        >>> states.sp
        sp(0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        >>> states.wc
        wc(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    for k in range(con.nmbzones):
        if con.zonetype[k] != ILAKE:
            if (flu.rfc[k]+flu.sfc[k]) > 0.:
                sta.wc[k] += flu.tf[k]*flu.rfc[k]/(flu.rfc[k]+flu.sfc[k])
                sta.sp[k] += flu.tf[k]*flu.sfc[k]/(flu.rfc[k]+flu.sfc[k])
        else:
            sta.wc[k] = 0.
            sta.sp[k] = 0.