def _parse_myinfo(client, command, actor, args):
    """Parse MYINFO and update the Host object."""
    _, server, version, usermodes, channelmodes = args.split(None, 5)[:5]
    s = client.server
    s.host = server
    s.version = version
    s.user_modes = set(usermodes)
    s.channel_modes = set(channelmodes)