def _parse_part(client, command, actor, args):
    """Parse a PART and update channel states, then dispatch events.

    Note that two events are dispatched here:
        - PART, because a user parted the channel
        - MEMBERS, because the channel's members changed
    """
    actor = User(actor)
    channel, _, message = args.partition(' :')
    channel = client.server.get_channel(channel)
    channel.remove_user(actor)
    if actor.nick == client.user.nick:
        client.server.remove_channel(channel)
    client.dispatch_event("PART", actor, channel, message)
    if actor.nick != client.user.nick:
        client.dispatch_event("MEMBERS", channel)