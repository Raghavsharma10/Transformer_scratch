def calc_qpin_v1(self):
    """Calculate the input discharge portions of the different response
    functions.

    Required derived parameters:
      |Nmb|
      |MaxQ|
      |DiffQ|

    Required flux sequence:
      |QIn|

    Calculated flux sequences:
      |QPIn|

    Examples:

        Initialize an arma model with three different response functions:

        >>> from hydpy.models.arma import *
        >>> parameterstep()
        >>> derived.nmb = 3
        >>> derived.maxq.shape = 3
        >>> derived.diffq.shape = 2
        >>> fluxes.qpin.shape = 3

        Define the maximum discharge value of the respective response
        functions and their successive differences:

        >>> derived.maxq(0.0, 2.0, 6.0)
        >>> derived.diffq(2., 4.)

        The first six examples are performed for inflow values ranging from
        0 to 12 m³/s:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(
        ...     model, model.calc_qpin_v1,
        ...     last_example=6,
        ...     parseqs=(fluxes.qin, fluxes.qpin))
        >>> test.nexts.qin = 0., 1., 2., 4., 6., 12.
        >>> test()
        | ex. |  qin |           qpin |
        -------------------------------
        |   1 |  0.0 | 0.0  0.0   0.0 |
        |   2 |  1.0 | 1.0  0.0   0.0 |
        |   3 |  2.0 | 2.0  0.0   0.0 |
        |   4 |  4.0 | 2.0  2.0   0.0 |
        |   5 |  6.0 | 2.0  4.0   0.0 |
        |   6 | 12.0 | 2.0  4.0   6.0 |


        The following two additional examples are just supposed to
        demonstrate method |calc_qpin_v1| also functions properly if
        there is only one response function, wherefore total discharge
        does not need to be divided:

        >>> derived.nmb = 1
        >>> derived.maxq.shape = 1
        >>> derived.diffq.shape = 0
        >>> fluxes.qpin.shape = 1
        >>> derived.maxq(0.)

        >>> test = UnitTest(
        ...     model, model.calc_qpin_v1,
        ...     first_example=7, last_example=8,
        ...                 parseqs=(fluxes.qin,
        ...                          fluxes.qpin))
        >>> test.nexts.qin = 0., 12.
        >>> test()
        | ex. |  qin | qpin |
        ---------------------
        |   7 |  0.0 |  0.0 |
        |   8 | 12.0 | 12.0 |

    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for idx in range(der.nmb-1):
        if flu.qin < der.maxq[idx]:
            flu.qpin[idx] = 0.
        elif flu.qin < der.maxq[idx+1]:
            flu.qpin[idx] = flu.qin-der.maxq[idx]
        else:
            flu.qpin[idx] = der.diffq[idx]
    flu.qpin[der.nmb-1] = max(flu.qin-der.maxq[der.nmb-1], 0.)