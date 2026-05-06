def calc_qout_v1(self):
    """Sum up the results of the different response functions.

    Required derived parameter:
      |Nmb|

    Required flux sequences:
      |QPOut|

    Calculated flux sequence:
      |QOut|

    Examples:

        Initialize an arma model with three different response functions:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(3)
        >>> fluxes.qpout.shape = 3

        Define the output values of the three response functions and
        apply method |calc_qout_v1|:

        >>> fluxes.qpout = 1.0, 2.0, 3.0
        >>> model.calc_qout_v1()
        >>> fluxes.qout
        qout(6.0)
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.qout = 0.
    for idx in range(der.nmb):
        flu.qout += flu.qpout[idx]