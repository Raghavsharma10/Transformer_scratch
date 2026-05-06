def callLater(self, when, what, *a, **kw):
    """
    Copied from twisted.internet.task.Clock, r20480.  Fixes the bug
    where the wrong DelayedCall would sometimes be returned.
    """
    dc =  base.DelayedCall(self.seconds() + when,
                           what, a, kw,
                           self.calls.remove,
                           lambda c: None,
                           self.seconds)
    self.calls.append(dc)
    self.calls.sort(lambda a, b: cmp(a.getTime(), b.getTime()))
    return dc