def calc_outflow_v1(self):
    """Calculate the total outflow of the dam.

    Note that the maximum function is used to prevent from negative outflow
    values, which could otherwise occur within the required level of
    numerical accuracy.

    Required flux sequences:
      |ActualRelease|
      |FloodDischarge|

    Calculated flux sequence:
      |Outflow|

    Basic equation:
      :math:`Outflow = max(ActualRelease + FloodDischarge, 0.)`

    Example:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> fluxes.actualrelease = 2.0
        >>> fluxes.flooddischarge = 3.0
        >>> model.calc_outflow_v1()
        >>> fluxes.outflow
        outflow(5.0)
        >>> fluxes.flooddischarge = -3.0
        >>> model.calc_outflow_v1()
        >>> fluxes.outflow
        outflow(0.0)
    """
    flu = self.sequences.fluxes.fastaccess
    flu.outflow = max(flu.actualrelease + flu.flooddischarge, 0.)