def add_serverconnection_methods(cls):
    """Add a bunch of methods to an :class:`irc.client.SimpleIRCClient`
    to send commands and messages.

    Basically it wraps a bunch of methdos from
    :class:`irc.client.ServerConnection` to be
    :meth:`irc.schedule.IScheduler.execute_after`.
    That way, you can easily send, even if the IRCClient is running in
    :class:`IRCClient.process_forever` in another thread.

    On the plus side you can use positional and keyword arguments instead of just positional ones.

    :param cls: The class to add the methods do.
    :type cls: :class:`irc.client.SimpleIRCClient`
    :returns: None
    """
    methods = ['action', 'admin', 'cap', 'ctcp', 'ctcp_reply',
               'globops', 'info', 'invite', 'ison', 'join',
               'kick', 'links', 'list', 'lusers', 'mode',
               'motd', 'names', 'nick', 'notice', 'oper', 'part',
               'part', 'pass_', 'ping', 'pong', 'privmsg',
               'privmsg_many', 'quit', 'send_raw', 'squit',
               'stats', 'time', 'topic', 'trace', 'user', 'userhost',
               'users', 'version', 'wallops', 'who', 'whois', 'whowas']
    for m in methods:
        method = _wrap_execute_after(m)
        f = getattr(irc.client.ServerConnection, m)
        method.__doc__ = f.__doc__
        setattr(cls, method.__name__, method)
    return cls