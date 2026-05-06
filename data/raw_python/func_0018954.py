def calc_qpout_v1(self):
    """Calculate the ARMA results for the different response functions.

    Required derived parameter:
      |Nmb|

    Required flux sequences:
      |QMA|
      |QAR|

    Calculated flux sequence:
      |QPOut|

    Examples:

        Initialize an arma model with three different response functions:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb(3)
        >>> fluxes.qma.shape = 3
        >>> fluxes.qar.shape = 3
        >>> fluxes.qpout.shape = 3

        Define the output values of the MA and of the AR processes
        associated with the three response functions and apply
        method |calc_qpout_v1|:

        >>> fluxes.qar = 4.0, 5.0, 6.0
        >>> fluxes.qma = 1.0, 2.0, 3.0
        >>> model.calc_qpout_v1()
        >>> fluxes.qpout
        qpout(5.0, 7.0, 9.0)
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for idx in range(der.nmb):
        flu.qpout[idx] = flu.qma[idx]+flu.qar[idx]