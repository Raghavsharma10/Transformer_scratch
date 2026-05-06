def calc_qref_v1(self):
    """Determine the reference discharge within the given space-time interval.

    Required state sequences:
      |QZ|
      |QA|

    Calculated flux sequence:
      |QRef|

    Basic equation:
      :math:`QRef = \\frac{QZ_{new}+QZ_{old}+QA_{old}}{3}`

    Example:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> states.qz.new = 3.0
        >>> states.qz.old = 2.0
        >>> states.qa.old = 1.0
        >>> model.calc_qref_v1()
        >>> fluxes.qref
        qref(2.0)
    """
    new = self.sequences.states.fastaccess_new
    old = self.sequences.states.fastaccess_old
    flu = self.sequences.fluxes.fastaccess
    flu.qref = (new.qz+old.qz+old.qa)/3.