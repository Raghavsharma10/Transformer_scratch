def _parse_msg(client, command, actor, args):
    """Parse a PRIVMSG or NOTICE and dispatch the corresponding event."""
    recipient, _, message = args.partition(' :')
    chantypes = client.server.features.get("CHANTYPES", "#")
    if recipient[0] in chantypes:
        recipient = client.server.get_channel(recipient) or recipient.lower()
    else:
        recipient = User(recipient)
    client.dispatch_event(command, actor, recipient, message)