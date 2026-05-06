def calc_qdgz1_qdgz2_v1(self):
    """Seperate total direct flow into a small and a fast component.

    Required control parameters:
      |A1|
      |A2|

    Required flux sequence:
      |QDGZ|

    Calculated state sequences:
      |QDGZ1|
      |QDGZ2|

    Basic equation:
       :math:`QDGZ2 = \\frac{(QDGZ-A2)^2}{QDGZ+A1-A2}`
       :math:`QDGZ1 = QDGZ - QDGZ1`

    Examples:

        The formula for calculating the amount of the fast component of
        direct flow is borrowed from the famous curve number approach.
        Parameter |A2| would be the initial loss and parameter |A1| the
        maximum storage, but one should not take this analogy too serious.
        Instead, with the value of parameter |A1| set to zero, parameter
        |A2| just defines the maximum amount of "slow" direct runoff per
        time step:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> a1(0.0)

        Let us set the value of |A2| to 4 mm/d, which is 2 mm/12h with
        respect to the selected simulation step size:

        >>> a2(4.0)
        >>> a2
        a2(4.0)
        >>> a2.value
        2.0

        Define a test function and let it calculate |QDGZ1| and |QDGZ1| for
        values of |QDGZ| ranging from -10 to 100 mm/12h:

        >>> from hydpy import UnitTest
        >>> test = UnitTest(model,
        ...                 model.calc_qdgz1_qdgz2_v1,
        ...                 last_example=6,
        ...                 parseqs=(fluxes.qdgz,
        ...                          states.qdgz1,
        ...                          states.qdgz2))
        >>> test.nexts.qdgz = -10.0, 0.0, 1.0, 2.0, 3.0, 100.0
        >>> test()
        | ex. |  qdgz | qdgz1 | qdgz2 |
        -------------------------------
        |   1 | -10.0 | -10.0 |   0.0 |
        |   2 |   0.0 |   0.0 |   0.0 |
        |   3 |   1.0 |   1.0 |   0.0 |
        |   4 |   2.0 |   2.0 |   0.0 |
        |   5 |   3.0 |   2.0 |   1.0 |
        |   6 | 100.0 |   2.0 |  98.0 |

        Setting |A2| to zero and |A1| to 4 mm/d (or 2 mm/12h) results in
        a smoother transition:

        >>> a2(0.0)
        >>> a1(4.0)
        >>> test()
        | ex. |  qdgz |    qdgz1 |     qdgz2 |
        --------------------------------------
        |   1 | -10.0 |    -10.0 |       0.0 |
        |   2 |   0.0 |      0.0 |       0.0 |
        |   3 |   1.0 | 0.666667 |  0.333333 |
        |   4 |   2.0 |      1.0 |       1.0 |
        |   5 |   3.0 |      1.2 |       1.8 |
        |   6 | 100.0 | 1.960784 | 98.039216 |

        Alternatively, one can mix these two configurations by setting
        the values of both parameters to 2 mm/h:

        >>> a2(2.0)
        >>> a1(2.0)
        >>> test()
        | ex. |  qdgz |    qdgz1 |    qdgz2 |
        -------------------------------------
        |   1 | -10.0 |    -10.0 |      0.0 |
        |   2 |   0.0 |      0.0 |      0.0 |
        |   3 |   1.0 |      1.0 |      0.0 |
        |   4 |   2.0 |      1.5 |      0.5 |
        |   5 |   3.0 | 1.666667 | 1.333333 |
        |   6 | 100.0 |     1.99 |    98.01 |

        Note the similarity of the results for very high values of total
        direct flow |QDGZ| in all three examples, which converge to the sum
        of the values of parameter |A1| and |A2|, representing the maximum
        value of `slow` direct flow generation per simulation step
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    sta = self.sequences.states.fastaccess
    if flu.qdgz > con.a2:
        sta.qdgz2 = (flu.qdgz-con.a2)**2/(flu.qdgz+con.a1-con.a2)
        sta.qdgz1 = flu.qdgz-sta.qdgz2
    else:
        sta.qdgz2 = 0.
        sta.qdgz1 = flu.qdgz