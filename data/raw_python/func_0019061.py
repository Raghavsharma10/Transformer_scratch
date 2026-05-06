def calc_tc_v1(self):
    """Adjust the measured air temperature to the altitude of the
    individual zones.

    Required control parameters:
      |NmbZones|
      |TCAlt|
      |ZoneZ|
      |ZRelT|

    Required input sequence:
      |hland_inputs.T|

    Calculated flux sequences:
      |TC|

    Basic equation:
      :math:`TC = T - TCAlt \\cdot (ZoneZ-ZRelT)`

    Examples:

        Prepare two zones, the first one lying at the reference
        height and the second one 200 meters above:

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(2)
        >>> zrelt(2.0)
        >>> zonez(2.0, 4.0)

        Applying the usual temperature lapse rate of 0.6°C/100m does
        not affect the temperature of the first zone but reduces the
        temperature of the second zone by 1.2°C:

        >>> tcalt(0.6)
        >>> inputs.t = 5.0
        >>> model.calc_tc_v1()
        >>> fluxes.tc
        tc(5.0, 3.8)
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nmbzones):
        flu.tc[k] = inp.t-con.tcalt[k]*(con.zonez[k]-con.zrelt)