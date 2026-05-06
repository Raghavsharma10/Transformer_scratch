def _wrap_execute_after(funcname):
    """Warp the given method, so it gets executed by the reactor

    Wrap a method of :data:`IRCCLient.out_connection`.

    The returned function should be assigned to a :class:`irc.client.SimpleIRCClient` class.

    :param funcname: the name of a :class:`irc.client.ServerConnection` method
    :type funcname: :class:`str`
    :returns: a new function, that executes the given one via :class:`irc.schedule.IScheduler.execute_after`
    :raises: None
    """
    def method(self, *args, **kwargs):
        f = getattr(self.out_connection, funcname)
        p = functools.partial(f, *args, **kwargs)
        self.reactor.scheduler.execute_after(0, p)
    method.__name__ = funcname
    return method