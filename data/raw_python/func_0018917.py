def calc_qbga_v1(self):
    """Perform the runoff concentration calculation for base flow.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required derived parameter:
      |KB|

    Required flux sequence:
      |QBGZ|

    Calculated state sequence:
      |QBGA|

    Basic equation:
       :math:`QBGA_{neu} = QBGA_{alt} +
       (QBGZ_{alt}-QBGA_{alt}) \\cdot (1-exp(-KB^{-1})) +
       (QBGZ_{neu}-QBGZ_{alt}) \\cdot (1-KB\\cdot(1-exp(-KB^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lland import *
        >>> parameterstep()
        >>> derived.kb(0.1)
        >>> states.qbgz.old = 2.0
        >>> states.qbgz.new = 4.0
        >>> states.qbga.old = 3.0
        >>> model.calc_qbga_v1()
        >>> states.qbga
        qbga(3.800054)

        First extreme test case (zero division is circumvented):

        >>> derived.kb(0.0)
        >>> model.calc_qbga_v1()
        >>> states.qbga
        qbga(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> derived.kb(1e500)
        >>> model.calc_qbga_v1()
        >>> states.qbga
        qbga(5.0)
    """
    der = self.parameters.derived.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    if der.kb <= 0.:
        new.qbga = new.qbgz
    elif der.kb > 1e200:
        new.qbga = old.qbga+new.qbgz-old.qbgz
    else:
        d_temp = (1.-modelutils.exp(-1./der.kb))
        new.qbga = (old.qbga +
                    (old.qbgz-old.qbga)*d_temp +
                    (new.qbgz-old.qbgz)*(1.-der.kb*d_temp))