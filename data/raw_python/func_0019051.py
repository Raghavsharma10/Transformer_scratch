def calc_av_uv_v1(self):
    """Calculate the flown through area and the wetted perimeter of both
    forelands.

    Note that the each foreland lies between the main channel and one
    outer embankment and that water flowing exactly above the a foreland
    is contributing to |AV|.  The theoretical surface seperating water
    above the main channel from water above the foreland is not
    contributing to |UV|, but the surface seperating water above the
    foreland from water above its outer embankment is contributing to |UV|.

    Required control parameters:
      |HM|
      |BV|
      |BNV|

    Required derived parameter:
      |HV|

    Required flux sequence:
      |H|

    Calculated flux sequence:
      |AV|
      |UV|

    Examples:

        Generally, right trapezoids are assumed.  Here, for simplicity, both
        forelands are assumed to be symmetrical.  Their smaller bases (bottoms)
        hava a length of 2 meters, their non-vertical legs show an inclination
        of 1 meter per 4 meters, and their height (depths) is 1 meter.  Both
        forelands lie 1 meter above the main channels bottom.

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> hm(1.0)
        >>> bv(2.0)
        >>> bnv(4.0)
        >>> derived.hv(1.0)

        The first example deals with normal flow conditions, where water flows
        within the main channel completely (|H| < |HM|):

        >>> fluxes.h = 0.5
        >>> model.calc_av_uv_v1()
        >>> fluxes.av
        av(0.0, 0.0)
        >>> fluxes.uv
        uv(0.0, 0.0)

        The second example deals with moderate high flow conditions, where
        water flows over both forelands, but not over their embankments
        (|HM| < |H| < (|HM| + |HV|)):

        >>> fluxes.h = 1.5
        >>> model.calc_av_uv_v1()
        >>> fluxes.av
        av(1.5, 1.5)
        >>> fluxes.uv
        uv(4.061553, 4.061553)

        The third example deals with extreme high flow conditions, where
        water flows over the both foreland and their outer embankments
        ((|HM| + |HV|) < |H|):

        >>> fluxes.h = 2.5
        >>> model.calc_av_uv_v1()
        >>> fluxes.av
        av(7.0, 7.0)
        >>> fluxes.uv
        uv(6.623106, 6.623106)

        The forth example assures that zero widths or hights of the forelands
        are handled properly:

        >>> bv.left = 0.0
        >>> derived.hv.right = 0.0
        >>> model.calc_av_uv_v1()
        >>> fluxes.av
        av(4.0, 3.0)
        >>> fluxes.uv
        uv(4.623106, 3.5)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for i in range(2):
        if flu.h <= con.hm:
            flu.av[i] = 0.
            flu.uv[i] = 0.
        elif flu.h <= (con.hm+der.hv[i]):
            flu.av[i] = (flu.h-con.hm)*(con.bv[i]+(flu.h-con.hm)*con.bnv[i]/2.)
            flu.uv[i] = con.bv[i]+(flu.h-con.hm)*(1.+con.bnv[i]**2)**.5
        else:
            flu.av[i] = (der.hv[i]*(con.bv[i]+der.hv[i]*con.bnv[i]/2.) +
                         ((flu.h-(con.hm+der.hv[i])) *
                          (con.bv[i]+der.hv[i]*con.bnv[i])))
            flu.uv[i] = ((con.bv[i])+(der.hv[i]*(1.+con.bnv[i]**2)**.5) +
                         (flu.h-(con.hm+der.hv[i])))