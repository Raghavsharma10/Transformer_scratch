def upgradeParentHook1to2(oldHook):
    """
    Add the scheduler attribute to the given L{_SubSchedulerParentHook}.
    """
    newHook = oldHook.upgradeVersion(
        oldHook.typeName, 1, 2,
        loginAccount=oldHook.loginAccount,
        scheduledAt=oldHook.scheduledAt,
        scheduler=oldHook.store.findFirst(Scheduler))
    return newHook