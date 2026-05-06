def _parse_topic(client, command, actor, args):
    """Parse a TOPIC and update channel state, then dispatch a TOPIC event."""
    channel, _, topic = args.partition(" :")
    channel = client.server.get_channel(channel)
    channel.topic = topic or None
    if actor:
        actor = User(actor)
    client.dispatch_event("TOPIC", actor, channel, topic)