def solve_dv_dt_v1(self):
    """Solve the differential equation of HydPy-L.

    At the moment, HydPy-L only implements a simple numerical solution of
    its underlying ordinary differential equation.  To increase the accuracy
    (or sometimes even to prevent instability) of this approximation, one
    can set the value of parameter |MaxDT| to a value smaller than the actual
    simulation step size.  Method |solve_dv_dt_v1| then applies the methods
    related to the numerical approximation multiple times and aggregates
    the results.

    Note that the order of convergence is one only.  It is hard to tell how
    short the internal simulation step needs to be to ensure a certain degree
    of accuracy.  In most cases one hour or very often even one day should be
    sufficient to gain acceptable results.  However, this strongly depends on
    the given water stage-volume-discharge relationship.  Hence it seems
    advisable to always define a few test waves and apply the llake model with
    different |MaxDT| values.  Afterwards, select a |MaxDT| value  lower than
    one which results in acceptable approximations for all test waves.  The
    computation time of the llake mode per substep is rather small, so always
    include a safety factor.

    Of course, an adaptive step size determination would be much more
    convenient...

    Required derived parameter:
      |NmbSubsteps|

    Used aide sequence:
      |llake_aides.V|
      |llake_aides.QA|

    Updated state sequence:
      |llake_states.V|

    Calculated flux sequence:
      |llake_fluxes.QA|

    Note that method |solve_dv_dt_v1| calls the versions of `calc_vq`,
    `interp_qa` and `calc_v_qa` selected by the respective application model.
    Hence, also their parameter and sequence specifications need to be
    considered.

    Basic equation:
      :math:`\\frac{dV}{dt}= QZ - QA(V)`
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    aid = self.sequences.aides.fastaccess
    flu.qa = 0.
    aid.v = old.v
    for _ in range(der.nmbsubsteps):
        self.calc_vq()
        self.interp_qa()
        self.calc_v_qa()
        flu.qa += aid.qa
    flu.qa /= der.nmbsubsteps
    new.v = aid.v