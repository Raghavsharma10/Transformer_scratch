def _parse_endofnames(client, command, actor, args):
    """Parse an ENDOFNAMES and dispatch a NAMES event for the channel."""
    args = args.split(" :", 1)[0] # Strip off human-readable message
    _, _, channel = args.rpartition(' ')
    channel = client.server.get_channel(channel) or channel.lower()
    client.dispatch_event('MEMBERS', channel)