def calc_nkor_v1(self):
    """Adjust the given precipitation values.

    Required control parameters:
      |NHRU|
      |KG|

    Required input sequence:
      |Nied|

    Calculated flux sequence:
      |NKor|

    Basic equation:
      :math:`NKor = KG \\cdot Nied`

    Example:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(3)
        >>> kg(0.8, 1.0, 1.2)
        >>> inputs.nied = 10.0
        >>> model.calc_nkor_v1()
        >>> fluxes.nkor
        nkor(8.0, 10.0, 12.0)
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        flu.nkor[k] = con.kg[k] * inp.nied