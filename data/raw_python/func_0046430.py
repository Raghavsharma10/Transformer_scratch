def _parse_kick(client, command, actor, args):
    """Parse a KICK and update channel states, then dispatch events.

    Note that two events are dispatched here:
        - KICK, because a user was kicked from the channel
        - MEMBERS, because the channel's members changed
    """
    actor = User(actor)
    args, _, message = args.partition(' :')
    channel, target = args.split()
    channel = client.server.get_channel(channel)
    channel.remove_user(target)
    target = User(target)
    if target.nick == client.user.nick:
        client.server.remove_channel(channel)
    client.dispatch_event("KICK", actor, target, channel, message)
    client.dispatch_event("MEMBERS", channel)