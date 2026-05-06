def _parse_join(client, command, actor, args):
    """Parse a JOIN and update channel states, then dispatch events.

    Note that two events are dispatched here:
        - JOIN, because a user joined the channel
        - MEMBERS, because the channel's members changed
    """
    actor = User(actor)
    channel = args.lstrip(' :').lower()
    if actor.nick == client.user.nick:
        client.server.add_channel(channel)
        client.user.host = actor.host # now we know our host per the server
    channel = client.server.get_channel(channel)
    channel.add_user(actor)
    client.dispatch_event("JOIN", actor, channel)
    if actor.nick != client.user.nick:
        # If this is us joining, the namreply will trigger this instead
        client.dispatch_event("MEMBERS", channel)