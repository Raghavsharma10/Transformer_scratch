def calc_evpo_v1(self):
    """Calculate land use and month specific values of potential
    evapotranspiration.

    Required control parameters:
      |NHRU|
      |Lnk|
      |FLn|

    Required derived parameter:
      |MOY|

    Required flux sequence:
      |ET0|

    Calculated flux sequence:
      |EvPo|

    Additional requirements:
      |Model.idx_sim|

    Basic equation:
      :math:`EvPo = FLn \\cdot ET0`

    Example:

        For clarity, this is more of a kind of an integration example.
        Parameter |FLn| both depends on time (the actual month) and space
        (the actual land use).  Firstly, let us define a initialization
        time period spanning the transition from June to July:

        >>> from hydpy import pub
        >>> pub.timegrids = '30.06.2000', '02.07.2000', '1d'

        Secondly, assume that the considered subbasin is differenciated in
        two HRUs, one of primarily consisting of arable land and the other
        one of deciduous forests:

        >>> from hydpy.models.lland import *
        >>> parameterstep('1d')
        >>> nhru(2)
        >>> lnk(ACKER, LAUBW)

        Thirdly, set the |FLn|
        values, one for the relevant months and land use classes:

        >>> fln.acker_jun = 1.299
        >>> fln.acker_jul = 1.304
        >>> fln.laubw_jun = 1.350
        >>> fln.laubw_jul = 1.365

        Fourthly, the index array connecting the simulation time steps
        defined above and the month indexes (0...11) can be retrieved
        from the |pub| module.  This can be done manually more
        conveniently via its update method:

        >>> derived.moy.update()
        >>> derived.moy
        moy(5, 6)

        Finally, the actual method (with its simple equation) is applied
        as usual:

        >>> fluxes.et0 = 2.0
        >>> model.idx_sim = 0
        >>> model.calc_evpo_v1()
        >>> fluxes.evpo
        evpo(2.598, 2.7)
        >>> model.idx_sim = 1
        >>> model.calc_evpo_v1()
        >>> fluxes.evpo
        evpo(2.608, 2.73)

        Reset module |pub| to not interfere the following examples:

        >>> del pub.timegrids
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for k in range(con.nhru):
        flu.evpo[k] = con.fln[con.lnk[k]-1, der.moy[self.idx_sim]] * flu.et0[k]