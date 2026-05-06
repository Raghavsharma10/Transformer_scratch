def calc_qa_v1(self):
    """Calculate outflow.

    The working equation is the analytical solution of the linear storage
    equation under the assumption of constant change in inflow during
    the simulation time step.

    Required flux sequence:
      |RK|

    Required state sequence:
      |QZ|

    Updated state sequence:
      |QA|

    Basic equation:
       :math:`QA_{neu} = QA_{alt} +
       (QZ_{alt}-QA_{alt}) \\cdot (1-exp(-RK^{-1})) +
       (QZ_{neu}-QZ_{alt}) \\cdot (1-RK\\cdot(1-exp(-RK^{-1})))`

    Examples:

        A normal test case:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> fluxes.rk(0.1)
        >>> states.qz.old = 2.0
        >>> states.qz.new = 4.0
        >>> states.qa.old = 3.0
        >>> model.calc_qa_v1()
        >>> states.qa
        qa(3.800054)

        First extreme test case (zero division is circumvented):

        >>> fluxes.rk(0.0)
        >>> model.calc_qa_v1()
        >>> states.qa
        qa(4.0)

        Second extreme test case (numerical overflow is circumvented):

        >>> fluxes.rk(1e201)
        >>> model.calc_qa_v1()
        >>> states.qa
        qa(5.0)
    """
    flu = self.sequences.fluxes.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    aid = self.sequences.aides.fastaccess
    if flu.rk <= 0.:
        new.qa = new.qz
    elif flu.rk > 1e200:
        new.qa = old.qa+new.qz-old.qz
    else:
        aid.temp = (1.-modelutils.exp(-1./flu.rk))
        new.qa = (old.qa +
                  (old.qz-old.qa)*aid.temp +
                  (new.qz-old.qz)*(1.-flu.rk*aid.temp))