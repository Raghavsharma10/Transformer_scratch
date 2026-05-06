def _parse_invite(client, command, actor, args):
    """Parse an INVITE and dispatch an event."""
    target, _, channel = args.rpartition(" ")
    client.dispatch_event("INVITE", actor, target, channel.lower())