def _parse_nick(client, command, actor, args):
    """Parse a NICK response, update state, and dispatch events.

    Note: this function dispatches both a NICK event and also one or more
    MEMBERS events for each channel the user that changed nick was in.
    """
    old_nick, _, _ = actor.partition('!')
    new_nick = args

    if old_nick == client.user.nick:
        client.user.nick = new_nick

    modified_channels = set()
    for channel in client.server.channels.itervalues():
        user = channel.members.get(old_nick)
        if user:
            user.nick = new_nick
            channel.members[new_nick] = user
            del channel.members[old_nick]
            modified_channels.add(channel.name)

    client.dispatch_event("NICK", old_nick, new_nick)
    for channel in modified_channels:
        client.dispatch_event("MEMBERS", channel)