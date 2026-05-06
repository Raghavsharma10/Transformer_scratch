def update_watervolume_v1(self):
    """Update the actual water volume.

    Required derived parameter:
      |Seconds|

    Required flux sequences:
      |Inflow|
      |Outflow|

    Updated state sequence:
      |WaterVolume|

    Basic equation:
      :math:`\\frac{d}{dt}WaterVolume = 1e-6 \\cdot (Inflow-Outflow)`

    Example:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> derived.seconds = 2e6
        >>> states.watervolume.old = 5.0
        >>> fluxes.inflow = 2.0
        >>> fluxes.outflow = 3.0
        >>> model.update_watervolume_v1()
        >>> states.watervolume
        watervolume(3.0)
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    old = self.sequences.states.fastaccess_old
    new = self.sequences.states.fastaccess_new
    new.watervolume = (old.watervolume +
                       der.seconds*(flu.inflow-flu.outflow)/1e6)