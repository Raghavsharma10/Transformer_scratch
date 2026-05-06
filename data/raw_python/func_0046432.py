def _parse_welcome(client, command, actor, args):
    """Parse a WELCOME and update user state, then dispatch a WELCOME event."""
    _, _, hostmask = args.rpartition(' ')
    client.user.update_from_hostmask(hostmask)
    client.dispatch_event("WELCOME", hostmask)