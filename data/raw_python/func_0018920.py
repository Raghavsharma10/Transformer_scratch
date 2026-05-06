def calc_qdga1_v1(self):
    """Perform the runoff concentration calculation for "slow" direct runoff.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required derived parameter:
      |KD1|

    Required state sequence:
      |QDGZ1|

    Calculated state sequence:
      |QDGA1|

    Basic equation:
       :math:`QDGA1_{neu} = QDGA1_{alt} +
       (QDGZ1_{alt}-QDGA1_{alt}) \\cdot (1-exp(-KD1^{-1})) +
       (QDGZ1_{neu}-QDGZ1_{alt}) \\cdot (1-KD1\\cdot(1-exp(-KD1^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> derived.kd1(0.1)
        >>> states.qdgz1.old = 2.0
        >>> states.qdgz1.new = 4.0
        >>> states.qdga1.old = 3.0
        >>> model.calc_qdga1_v1()
        >>> states.qdga1
        qdga1(3.800054)

        First extreme test case (zero division is circumvented):

        >>> derived.kd1(0.0)
        >>> model.calc_qdga1_v1()
        >>> states.qdga1
        qdga1(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> derived.kd1(1e500)
        >>> model.calc_qdga1_v1()
        >>> states.qdga1
        qdga1(5.0)
    """
    der = self.parameters.derived.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    if der.kd1 <= 0.:
        new.qdga1 = new.qdgz1
    elif der.kd1 > 1e200:
        new.qdga1 = old.qdga1+new.qdgz1-old.qdgz1
    else:
        d_temp = (1.-modelutils.exp(-1./der.kd1))
        new.qdga1 = (old.qdga1 +
                     (old.qdgz1-old.qdga1)*d_temp +
                     (new.qdgz1-old.qdgz1)*(1.-der.kd1*d_temp))