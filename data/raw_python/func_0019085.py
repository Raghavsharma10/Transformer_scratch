def calc_qt_v1(self):
    """Calculate the total discharge after possible abstractions.

    Required control parameter:
      |Abstr|

    Required flux sequence:
      |OutUH|

    Calculated flux sequence:
      |QT|

    Basic equation:
        :math:`QT = max(OutUH - Abstr, 0)`

    Examples:

        Trying to abstract less then available, as much as available and
        less then available results in:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> abstr(2.0)
        >>> fluxes.outuh = 2.0
        >>> model.calc_qt_v1()
        >>> fluxes.qt
        qt(1.0)
        >>> fluxes.outuh = 1.0
        >>> model.calc_qt_v1()
        >>> fluxes.qt
        qt(0.0)
        >>> fluxes.outuh = 0.5
        >>> model.calc_qt_v1()
        >>> fluxes.qt
        qt(0.0)

        Note that "negative abstractions" are allowed:

        >>> abstr(-2.0)
        >>> fluxes.outuh = 1.0
        >>> model.calc_qt_v1()
        >>> fluxes.qt
        qt(2.0)
    """
    con = self.parameters.control.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.qt = max(flu.outuh-con.abstr, 0.)