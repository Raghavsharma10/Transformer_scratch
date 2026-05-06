def calc_et0_wet0_v1(self):
    """Correct the given reference evapotranspiration and update the
    corresponding log sequence.

    Required control parameters:
      |NHRU|
      |KE|
      |WfET0|

    Required input sequence:
      |PET|

    Calculated flux sequence:
      |ET0|

    Updated log sequence:
      |WET0|

    Basic equations:
      :math:`ET0_{new} = WfET0 \\cdot KE \\cdot PET +
      (1-WfET0) \\cdot ET0_{alt}`

    Example:

        Prepare four hydrological response units with different value
        combinations of parameters |KE| and |WfET0|:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> simulationstep('12h')
        >>> nhru(4)
        >>> ke(0.8, 1.2, 0.8, 1.2)
        >>> wfet0(2.0, 2.0, 0.2, 0.2)

        Note that the actual value of time dependend parameter |WfET0|
        is reduced due the difference between the given parameter and
        simulation time steps:

        >>> from hydpy import round_
        >>> round_(wfet0.values)
        1.0, 1.0, 0.1, 0.1

        For the first two hydrological response units, the given |PET|
        value is modified by -0.4 mm and +0.4 mm, respectively.  For the
        other two response units, which weight the "new" evaporation
        value with 10 %, |ET0| does deviate from the old value of |WET0|
        by -0.04 mm and +0.04 mm only:

        >>> inputs.pet = 2.0
        >>> logs.wet0 = 2.0
        >>> model.calc_et0_wet0_v1()
        >>> fluxes.et0
        et0(1.6, 2.4, 1.96, 2.04)
        >>> logs.wet0
        wet0([[1.6, 2.4, 1.96, 2.04]])
    """
    con = self.parameters.control.fastaccess
    inp = self.sequences.inputs.fastaccess
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    for k in range(con.nhru):
        flu.et0[k] = (con.wfet0[k]*con.ke[k]*inp.pet +
                      (1.-con.wfet0[k])*log.wet0[0, k])
        log.wet0[0, k] = flu.et0[k]