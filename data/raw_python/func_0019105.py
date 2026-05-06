def corr_dw_v1(self):
    """Adjust the water stage drop to the highest value allowed and correct
    the associated fluxes.

    Note that method |corr_dw_v1| calls the method `interp_v` of the
    respective application model.  Hence the requirements of the actual
    `interp_v` need to be considered additionally.

    Required control parameter:
      |MaxDW|

    Required derived parameters:
      |llake_derived.TOY|
      |Seconds|

    Required flux sequence:
      |QZ|

    Updated flux sequence:
      |llake_fluxes.QA|

    Updated state sequences:
      |llake_states.W|
      |llake_states.V|

    Basic Restriction:
      :math:`W_{old} - W_{new} \\leq MaxDW`

    Examples:

        In preparation for the following examples, define a short simulation
        time period with a simulation step size of 12 hours and initialize
        the required model object:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000.01.01', '2000.01.04', '12h'
        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> derived.toy.update()
        >>> derived.seconds.update()

        Select the first half of the second day of January as the simulation
        step relevant for the following examples:

        >>> model.idx_sim = pub.timegrids.init['2000.01.02']

        The following tests are based on method |interp_v_v1| for the
        interpolation of the stored water volume based on the corrected
        water stage:

        >>> model.interp_v = model.interp_v_v1

        For the sake of simplicity, the underlying `w`-`v` relationship is
        assumed to be linear:

        >>> n(2.)
        >>> w(0., 1.)
        >>> v(0., 1e6)

        The maximum drop in water stage for the first half of the second
        day of January is set to 0.4 m/d.  Note that, due to the difference
        between the parameter step size and the simulation step size, the
        actual value used for calculation is 0.2 m/12h:

        >>> maxdw(_1_1_18=.1,
        ...       _1_2_6=.4,
        ...       _1_2_18=.1)
        >>> maxdw
        maxdw(toy_1_1_18_0_0=0.1,
              toy_1_2_6_0_0=0.4,
              toy_1_2_18_0_0=0.1)
        >>> from hydpy import round_
        >>> round_(maxdw.value[2])
        0.2

        Define old and new water stages and volumes in agreement with the
        given linear relationship:

        >>> states.w.old = 1.
        >>> states.v.old = 1e6
        >>> states.w.new = .9
        >>> states.v.new = 9e5

        Also define an inflow and an outflow value.  Note the that the latter
        is set to zero, which is inconsistent with the actual water stage drop
        defined above, but done for didactic reasons:

        >>> fluxes.qz = 1.
        >>> fluxes.qa = 0.

        Calling the |corr_dw_v1| method does not change the values of
        either of following sequences, as the actual drop (0.1 m/12h) is
        smaller than the allowed drop (0.2 m/12h):

        >>> model.corr_dw_v1()
        >>> states.w
        w(0.9)
        >>> states.v
        v(900000.0)
        >>> fluxes.qa
        qa(0.0)

        Note that the values given above are not recalculated, which can
        clearly be seen for the lake outflow, which is still zero.

        Through setting the new value of the water stage to 0.6 m, the actual
        drop (0.4 m/12h) exceeds the allowed drop (0.2 m/12h). Hence the
        water stage is trimmed and the other values are recalculated:

        >>> states.w.new = .6
        >>> model.corr_dw_v1()
        >>> states.w
        w(0.8)
        >>> states.v
        v(800000.0)
        >>> fluxes.qa
        qa(5.62963)

        Through setting the maximum water stage drop to zero, method
        |corr_dw_v1| is effectively disabled.  Regardless of the actual
        change in water stage, no trimming or recalculating is performed:

        >>> maxdw.toy_01_02_06 = 0.
        >>> states.w.new = .6
        >>> model.corr_dw_v1()
        >>> states.w
        w(0.6)
        >>> states.v
        v(800000.0)
        >>> fluxes.qa
        qa(5.62963)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    idx = der.toy[self.idx_sim]
    if (con.maxdw[idx] > 0.) and ((old.w-new.w) > con.maxdw[idx]):
        new.w = old.w-con.maxdw[idx]
        self.interp_v()
        flu.qa = flu.qz+(old.v-new.v)/der.seconds