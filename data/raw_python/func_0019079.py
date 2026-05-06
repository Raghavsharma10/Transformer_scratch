def calc_q0_perc_uz_v1(self):
    """Perform the upper zone layer routine which determines percolation
    to the lower zone layer and the fast response of the hland model.
    Note that the system behaviour of this method depends strongly on the
    specifications of the options |RespArea| and |RecStep|.

    Required control parameters:
      |RecStep|
      |PercMax|
      |K|
      |Alpha|

    Required derived parameters:
      |DT|

    Required fluxes sequence:
      |InUZ|

    Calculated fluxes sequences:
      |Perc|
      |Q0|

    Updated state sequence:
      |UZ|

    Basic equations:
      :math:`\\frac{dUZ}{dt} = InUZ - Perc - Q0` \n
      :math:`Perc = PercMax \\cdot ContriArea` \n
      :math:`Q0 = K * \\cdot \\left( \\frac{UZ}{ContriArea} \\right)^{1+Alpha}`

    Examples:

        The upper zone layer routine is an exception compared to
        the other routines of the HydPy-H-Land model, regarding its
        consideration of numerical accuracy.  To increase the accuracy of
        the numerical integration of the underlying ordinary differential
        equation, each simulation step can be divided into substeps, which
        are all solved with first order accuracy.  In the first example,
        this option is omitted through setting the RecStep parameter to one:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> recstep(2)
        >>> derived.dt = 1/recstep
        >>> percmax(2.0)
        >>> alpha(1.0)
        >>> k(2.0)
        >>> fluxes.contriarea = 1.0
        >>> fluxes.inuz = 0.0
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(1.0)
        >>> fluxes.q0
        q0(0.0)
        >>> states.uz
        uz(0.0)

        Due to the sequential calculation of the upper zone routine, the
        upper zone storage is drained completely through percolation and
        no water is left for fast discharge response.  By dividing the
        simulation step in 100 substeps, the results are quite different:

        >>> recstep(200)
        >>> derived.dt = 1.0/recstep
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.786934)
        >>> fluxes.q0
        q0(0.213066)
        >>> states.uz
        uz(0.0)

        Note that the assumed length of the simulation step is only a
        half day. Hence the effective values of the maximum percolation
        rate and the storage coefficient is not 2 but 1:

        >>> percmax
        percmax(2.0)
        >>> k
        k(2.0)
        >>> percmax.value
        1.0
        >>> k.value
        1.0

        By decreasing the contributing area one decreases percolation but
        increases fast discharge response:

        >>> fluxes.contriarea = 0.5
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.434108)
        >>> fluxes.q0
        q0(0.565892)
        >>> states.uz
        uz(0.0)

        Resetting RecStep leads to more transparent results.  Note that, due
        to the large value of the storage coefficient and the low accuracy
        of the numerical approximation, direct discharge drains the rest of
        the upper zone storage:

        >>> recstep(2)
        >>> derived.dt = 1.0/recstep
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.5)
        >>> fluxes.q0
        q0(0.5)
        >>> states.uz
        uz(0.0)

        Applying a more reasonable storage coefficient results in:

        >>> k(0.5)
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.5)
        >>> fluxes.q0
        q0(0.25)
        >>> states.uz
        uz(0.25)

        Adding an input of 0.3 mm results the same percolation value (which,
        in the given example, is determined by the maximum percolation rate
        only), but in an increases value of the direct response (which
        always depends on the actual upper zone storage directly):

        >>> fluxes.inuz = 0.3
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.5)
        >>> fluxes.q0
        q0(0.64)
        >>> states.uz
        uz(0.16)

        Due to the same reasons, another increase in numerical accuracy has
        no impact on percolation but decreases the direct response in the
        given example:

        >>> recstep(200)
        >>> derived.dt = 1.0/recstep
        >>> states.uz = 1.0
        >>> model.calc_q0_perc_uz_v1()
        >>> fluxes.perc
        perc(0.5)
        >>> fluxes.q0
        q0(0.421708)
        >>> states.uz
        uz(0.378292)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    flu.perc = 0.
    flu.q0 = 0.
    for dummy in range(con.recstep):
        # First state update related to the upper zone input.
        sta.uz += der.dt*flu.inuz
        # Second state update related to percolation.
        d_perc = min(der.dt*con.percmax*flu.contriarea, sta.uz)
        sta.uz -= d_perc
        flu.perc += d_perc
        # Third state update related to fast runoff response.
        if sta.uz > 0.:
            if flu.contriarea > 0.:
                d_q0 = (der.dt*con.k *
                        (sta.uz/flu.contriarea)**(1.+con.alpha))
                d_q0 = min(d_q0, sta.uz)
            else:
                d_q0 = sta.uz
            sta.uz -= d_q0
            flu.q0 += d_q0
        else:
            d_q0 = 0.