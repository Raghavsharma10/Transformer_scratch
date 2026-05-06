def calc_allowedremoterelieve_v1(self):
    """Get the allowed remote relieve of the last simulation step.

    Required log sequence:
      |LoggedAllowedRemoteRelieve|

    Calculated flux sequence:
      |AllowedRemoteRelieve|

    Basic equation:
      :math:`AllowedRemoteRelieve = LoggedAllowedRemoteRelieve`

    Example:

        >>> from hydpy.models.dam import *
        >>> parameterstep()
        >>> logs.loggedallowedremoterelieve = 2.0
        >>> model.calc_allowedremoterelieve_v1()
        >>> fluxes.allowedremoterelieve
        allowedremoterelieve(2.0)
    """
    flu = self.sequences.fluxes.fastaccess
    log = self.sequences.logs.fastaccess
    flu.allowedremoterelieve = log.loggedallowedremoterelieve[0]