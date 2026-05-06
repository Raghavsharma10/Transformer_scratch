def _schedulerServiceSpecialCase(empowered, pups):
    """
    This function creates (or returns a previously created) L{IScheduler}
    powerup.

    If L{IScheduler} powerups were found on C{empowered}, the first of those
    is given priority.  Otherwise, a site L{Store} or a user L{Store} will
    have any pre-existing L{IScheduler} powerup associated with them (on the
    hackish cache attribute C{_schedulerService}) returned, or a new one
    created if none exists already.
    """
    from axiom.scheduler import _SiteScheduler, _UserScheduler

    # Give precedence to anything found in the store
    for pup in pups:
        return pup
    # If the empowered is a store, construct a scheduler for it.
    if isinstance(empowered, Store):
        if getattr(empowered, '_schedulerService', None) is None:
            if empowered.parent is None:
                sched = _SiteScheduler(empowered)
            else:
                sched = _UserScheduler(empowered)
            empowered._schedulerService = sched
        return empowered._schedulerService
    return None