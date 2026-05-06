def clockIsBroken():
    """
    Returns whether twisted.internet.task.Clock has the bug that
    returns the wrong DelayedCall or not.
    """
    clock = Clock()
    dc1 = clock.callLater(10, lambda: None)
    dc2 = clock.callLater(1, lambda: None)
    if dc1 is dc2:
        return True
    else:
        return False