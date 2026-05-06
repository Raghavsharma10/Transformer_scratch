def calc_requiredremoterelease_v2(self):
    """Get the required remote release of the last simulation step.

    Required log sequence:
      |LoggedRequiredRemoteRelease|

    Calculated flux sequence:
      |RequiredRemoteRelease|

    Basic equation:
      :math:`RequiredRemoteRelease = LoggedRequiredRemoteRelease`

    Example:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> logs.loggedrequiredremoterelease = 3.0
        >>> model.calc_requiredremoterelease_v2()
        >>> fluxes.requiredremoterelease
        requiredremoterelease(3.0)
    """
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    flu.requiredremoterelease = log.loggedrequiredremoterelease[0]