def _parse_quit(client, command, actor, args):
    """Parse a QUIT and update channel states, then dispatch events.

    Note that two events are dispatched here:
        - QUIT, because a user quit the server
        - MEMBERS, for each channel the user is no longer in
    """
    actor = User(actor)
    _, _, message = args.partition(':')
    client.dispatch_event("QUIT", actor, message)
    for chan in client.server.channels.itervalues():
        if actor.nick in chan.members:
            chan.remove_user(actor)
            client.dispatch_event("MEMBERS", chan)