def calc_qiga2_v1(self):
    """Perform the runoff concentration calculation for the second
    interflow component.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required derived parameter:
      |KI2|

    Required state sequence:
      |QIGZ2|

    Calculated state sequence:
      |QIGA2|

    Basic equation:
       :math:`QIGA2_{neu} = QIGA2_{alt} +
       (QIGZ2_{alt}-QIGA2_{alt}) \\cdot (1-exp(-KI2^{-1})) +
       (QIGZ2_{neu}-QIGZ2_{alt}) \\cdot (1-KI2\\cdot(1-exp(-KI2^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> derived.ki2(0.1)
        >>> states.qigz2.old = 2.0
        >>> states.qigz2.new = 4.0
        >>> states.qiga2.old = 3.0
        >>> model.calc_qiga2_v1()
        >>> states.qiga2
        qiga2(3.800054)

        First extreme test case (zero division is circumvented):

        >>> derived.ki2(0.0)
        >>> model.calc_qiga2_v1()
        >>> states.qiga2
        qiga2(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> derived.ki2(1e500)
        >>> model.calc_qiga2_v1()
        >>> states.qiga2
        qiga2(5.0)
    """
    der = self.parameters.derived.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    if der.ki2 <= 0.:
        new.qiga2 = new.qigz2
    elif der.ki2 > 1e200:
        new.qiga2 = old.qiga2+new.qigz2-old.qigz2
    else:
        d_temp = (1.-modelutils.exp(-1./der.ki2))
        new.qiga2 = (old.qiga2 +
                     (old.qigz2-old.qiga2)*d_temp +
                     (new.qigz2-old.qigz2)*(1.-der.ki2*d_temp))