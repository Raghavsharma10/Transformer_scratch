def calc_vq_v1(self):
    """Calculate the auxiliary term.

    Required derived parameters:
      |Seconds|
      |NmbSubsteps|

    Required flux sequence:
      |QZ|

    Required aide sequence:
      |llake_aides.V|

    Calculated aide sequence:
      |llake_aides.VQ|

    Basic equation:
      :math:`VQ = 2 \\cdot V + \\frac{Seconds}{NmbSubsteps} \\cdot QZ`

    Example:

        The following example shows that the auxiliary term `vq` does not
        depend on the (outer) simulation step size but on the (inner)
        calculation step size defined by parameter `maxdt`:

        >>> from hydpy.models.llake import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> maxdt('6h')
        >>> derived.seconds.update()
        >>> derived.nmbsubsteps.update()
        >>> fluxes.qz = 2.
        >>> aides.v = 1e5
        >>> model.calc_vq_v1()
        >>> aides.vq
        vq(243200.0)
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    aid = self.sequences.aides.fastaccess
    aid.vq = 2.*aid.v+der.seconds/der.nmbsubsteps*flu.qz