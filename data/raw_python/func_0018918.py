def calc_qiga1_v1(self):
    """Perform the runoff concentration calculation for the first
    interflow component.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required derived parameter:
      |KI1|

    Required state sequence:
      |QIGZ1|

    Calculated state sequence:
      |QIGA1|

    Basic equation:
       :math:`QIGA1_{neu} = QIGA1_{alt} +
       (QIGZ1_{alt}-QIGA1_{alt}) \\cdot (1-exp(-KI1^{-1})) +
       (QIGZ1_{neu}-QIGZ1_{alt}) \\cdot (1-KI1\\cdot(1-exp(-KI1^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> derived.ki1(0.1)
        >>> states.qigz1.old = 2.0
        >>> states.qigz1.new = 4.0
        >>> states.qiga1.old = 3.0
        >>> model.calc_qiga1_v1()
        >>> states.qiga1
        qiga1(3.800054)

        First extreme test case (zero division is circumvented):

        >>> derived.ki1(0.0)
        >>> model.calc_qiga1_v1()
        >>> states.qiga1
        qiga1(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> derived.ki1(1e500)
        >>> model.calc_qiga1_v1()
        >>> states.qiga1
        qiga1(5.0)
    """
    der = self.parameters.derived.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    if der.ki1 <= 0.:
        new.qiga1 = new.qigz1
    elif der.ki1 > 1e200:
        new.qiga1 = old.qiga1+new.qigz1-old.qigz1
    else:
        d_temp = (1.-modelutils.exp(-1./der.ki1))
        new.qiga1 = (old.qiga1 +
                     (old.qigz1-old.qiga1)*d_temp +
                     (new.qigz1-old.qigz1)*(1.-der.ki1*d_temp))