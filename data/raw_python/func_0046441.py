def _parse_nicknameinuse(client, command, actor, args):
    """Parse a NICKNAMEINUSE message and dispatch an event.

    The parameter passed along with the event is the nickname
    which is already in use.
    """
    nick, _, _ = args.rpartition(" ")
    client.dispatch_event("NICKNAMEINUSE", nick)