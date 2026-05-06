def calc_qdga2_v1(self):
    """Perform the runoff concentration calculation for "fast" direct runoff.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required derived parameter:
      |KD2|

    Required state sequence:
      |QDGZ2|

    Calculated state sequence:
      |QDGA2|

    Basic equation:
       :math:`QDGA2_{neu} = QDGA2_{alt} +
       (QDGZ2_{alt}-QDGA2_{alt}) \\cdot (1-exp(-KD2^{-1})) +
       (QDGZ2_{neu}-QDGZ2_{alt}) \\cdot (1-KD2\\cdot(1-exp(-KD2^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> derived.kd2(0.1)
        >>> states.qdgz2.old = 2.0
        >>> states.qdgz2.new = 4.0
        >>> states.qdga2.old = 3.0
        >>> model.calc_qdga2_v1()
        >>> states.qdga2
        qdga2(3.800054)

        First extreme test case (zero division is circumvented):

        >>> derived.kd2(0.0)
        >>> model.calc_qdga2_v1()
        >>> states.qdga2
        qdga2(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> derived.kd2(1e500)
        >>> model.calc_qdga2_v1()
        >>> states.qdga2
        qdga2(5.0)
    """
    der = self.parameters.derived.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    if der.kd2 <= 0.:
        new.qdga2 = new.qdgz2
    elif der.kd2 > 1e200:
        new.qdga2 = old.qdga2+new.qdgz2-old.qdgz2
    else:
        d_temp = (1.-modelutils.exp(-1./der.kd2))
        new.qdga2 = (old.qdga2 +
                     (old.qdgz2-old.qdga2)*d_temp +
                     (new.qdgz2-old.qdgz2)*(1.-der.kd2*d_temp))