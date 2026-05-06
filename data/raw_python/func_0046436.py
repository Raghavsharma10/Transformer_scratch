def _parse_namreply(client, command, actor, args):
    """Parse NAMREPLY and update a Channel object."""
    prefixes = client._get_prefixes()

    channelinfo, _, useritems = args.partition(' :')
    _, _, channel = channelinfo.rpartition(' ')  # channeltype channelname

    c = client.server.get_channel(channel)
    if not c:
        _log.warning("Ignoring NAMREPLY for channel '%s' which we are not in.",
            channel)
        return

    # We bypass Channel.add_user() here because we just want to sync in any
    # users we don't already have, regardless of if other users exist, and
    # we don't want the warning spam.
    for nick in useritems.split():
        modes = set()
        while nick[0] in prefixes:
            modes.add(prefixes[nick[0]])
            nick = nick[1:]
        user = c.members.get(nick)
        if not user:
            user = c.members[nick] = User(nick)
            _log.debug("Added user %s to channel %s", user, channel)
        user.modes |= modes