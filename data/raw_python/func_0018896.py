def calc_tkor_v1(self):
    """Adjust the given air temperature values.

    Required control parameters:
      |NHRU|
      |KT|

    Required input sequence:
      |TemL|

    Calculated flux sequence:
      |TKor|

    Basic equation:
      :math:`TKor = KT + TemL`

    Example:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(3)
        >>> kt(-2.0, 0.0, 2.0)
        >>> inputs.teml(1.)
        >>> model.calc_tkor_v1()
        >>> fluxes.tkor
        tkor(-1.0, 1.0, 3.0)
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        flu.tkor[k] = con.kt[k] + inp.teml