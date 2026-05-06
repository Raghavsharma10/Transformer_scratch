def calc_requiredrelease_v2(self):
    """Calculate the water release (immediately downstream) required for
    reducing drought events.

    Required control parameter:
      |NearDischargeMinimumThreshold|

    Required derived parameter:
      |dam_derived.TOY|

    Calculated flux sequence:
      |RequiredRelease|

    Basic equation:
      :math:`RequiredRelease = NearDischargeMinimumThreshold`

    Examples:

        As in the examples above, define a short simulation time period first:

        >>> from hydpy import pub
        >>> pub.timegrids = '2001.03.30', '2001.04.03', '1d'

        Prepare the dam model:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> derived.toy.update()

        Define a minimum discharge value for a cross section immediately
        downstream of 4 m³/s for the summer months and of 0 m³/s for the
        winter months:

        >>> neardischargeminimumthreshold(_11_1_12=0.0, _03_31_12=0.0,
        ...                               _04_1_12=4.0, _10_31_12=4.0)


        As to be expected, the calculated required release is 0.0 m³/s
        on May 31 and 4.0 m³/s on April 1:

        >>> model.idx_sim = pub.timegrids.init['2001.03.31']
        >>> model.calc_requiredrelease_v2()
        >>> fluxes.requiredrelease
        requiredrelease(0.0)
        >>> model.idx_sim = pub.timegrids.init['2001.04.01']
        >>> model.calc_requiredrelease_v2()
        >>> fluxes.requiredrelease
        requiredrelease(4.0)
    """
    con = self.parameters.control.fastaccess
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    flu.requiredrelease = con.neardischargeminimumthreshold[
        der.toy[self.idx_sim]]