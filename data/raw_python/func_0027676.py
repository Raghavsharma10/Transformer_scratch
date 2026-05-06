def upgradeProcessor1to2(oldProcessor):
    """
    Batch processors stopped polling at version 2, so they no longer needed the
    idleInterval attribute.  They also gained a scheduled attribute which
    tracks their interaction with the scheduler.  Since they stopped polling,
    we also set them up as a timed event here to make sure that they don't
    silently disappear, never to be seen again: running them with the scheduler
    gives them a chance to figure out what's up and set up whatever other state
    they need to continue to run.

    Since this introduces a new dependency of all batch processors on a powerup
    for the IScheduler, install a Scheduler or a SubScheduler if one is not
    already present.
    """
    newProcessor = oldProcessor.upgradeVersion(
        oldProcessor.typeName, 1, 2,
        busyInterval=oldProcessor.busyInterval)
    newProcessor.scheduled = extime.Time()

    s = newProcessor.store
    sch = iaxiom.IScheduler(s, None)
    if sch is None:
        if s.parent is None:
            # Only site stores have no parents.
            sch = Scheduler(store=s)
        else:
            # Substores get subschedulers.
            sch = SubScheduler(store=s)
        installOn(sch, s)

    # And set it up to run.
    sch.schedule(newProcessor, newProcessor.scheduled)
    return newProcessor