def calc_avr_uvr_v1(self):
    """Calculate the flown through area and the wetted perimeter of both
    outer embankments.

    Note that each outer embankment lies beyond its foreland and that all
    water flowing exactly above the a embankment is added to |AVR|.
    The theoretical surface seperating water above the foreland from water
    above its embankment is not contributing to |UVR|.

    Required control parameters:
      |HM|
      |BNVR|

    Required derived parameter:
      |HV|

    Required flux sequence:
      |H|

    Calculated flux sequence:
      |AVR|
      |UVR|

    Examples:

        Generally, right trapezoids are assumed.  Here, for simplicity, both
        forelands are assumed to be symmetrical.  Their smaller bases (bottoms)
        hava a length of 2 meters, their non-vertical legs show an inclination
        of 1 meter per 4 meters, and their height (depths) is 1 meter.  Both
        forelands lie 1 meter above the main channels bottom.

        Generally, a triangles are assumed, with the vertical side
        seperating the foreland from its outer embankment.  Here, for
        simplicity, both forelands are assumed to be symmetrical.  Their
        inclinations are 1 meter per 4 meters and their lowest point is
        1 meter above the forelands bottom and 2 meters above the main
        channels bottom:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> hm(1.0)
        >>> bnvr(4.0)
        >>> derived.hv(1.0)

        The first example deals with moderate high flow conditions, where
        water flows over the forelands, but not over their outer embankments
        (|HM| < |H| < (|HM| + |HV|)):

        >>> fluxes.h = 1.5
        >>> model.calc_avr_uvr_v1()
        >>> fluxes.avr
        avr(0.0, 0.0)
        >>> fluxes.uvr
        uvr(0.0, 0.0)

        The second example deals with extreme high flow conditions, where
        water flows over the both foreland and their outer embankments
        ((|HM| + |HV|) < |H|):

        >>> fluxes.h = 2.5
        >>> model.calc_avr_uvr_v1()
        >>> fluxes.avr
        avr(0.5, 0.5)
        >>> fluxes.uvr
        uvr(2.061553, 2.061553)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    for i in range(2):
        if flu.h <= (con.hm+der.hv[i]):
            flu.avr[i] = 0.
            flu.uvr[i] = 0.
        else:
            flu.avr[i] = (flu.h-(con.hm+der.hv[i]))**2*con.bnvr[i]/2.
            flu.uvr[i] = (flu.h-(con.hm+der.hv[i]))*(1.+con.bnvr[i]**2)**.5