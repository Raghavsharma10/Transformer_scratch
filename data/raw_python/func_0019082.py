def calc_q1_lz_v1(self):
    """Calculate the slow response of the lower zone layer.

    Required control parameters:
        |K4|
        |Gamma|

    Calculated fluxes sequence:
        |Q1|

    Updated state sequence:
        |LZ|

    Basic equations:
        :math:`\\frac{dLZ}{dt} = -Q1` \n
        :math:`Q1 = \\Bigl \\lbrace
        {
        {K4 \\cdot LZ^{1+Gamma} \\ | \\ LZ > 0}
        \\atop
        {0 \\ | \\ LZ\\leq 0}
        }`

    Examples:

        As long as the lower zone storage is negative...

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> k4(0.2)
        >>> gamma(0.0)
        >>> states.lz = -2.0
        >>> model.calc_q1_lz_v1()
        >>> fluxes.q1
        q1(0.0)
        >>> states.lz
        lz(-2.0)

        ...or zero, no slow discharge response occurs:

        >>> states.lz = 0.0
        >>> model.calc_q1_lz_v1()
        >>> fluxes.q1
        q1(0.0)
        >>> states.lz
        lz(0.0)

        For storage values above zero the linear...

        >>> states.lz = 2.0
        >>> model.calc_q1_lz_v1()
        >>> fluxes.q1
        q1(0.2)
        >>> states.lz
        lz(1.8)

        ...or nonlinear storage routing equation applies:

        >>> gamma(1.)
        >>> states.lz = 2.0
        >>> model.calc_q1_lz_v1()
        >>> fluxes.q1
        q1(0.4)
        >>> states.lz
        lz(1.6)

        Note that the assumed length of the simulation step is only a
        half day. Hence the effective value of the storage coefficient
        is not 0.2 but 0.1:

        >>> k4
        k4(0.2)
        >>> k4.value
        0.1
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    if sta.lz > 0.:
        flu.q1 = con.k4*sta.lz**(1.+con.gamma)
    else:
        flu.q1 = 0.
    sta.lz -= flu.q1