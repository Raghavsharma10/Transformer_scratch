def _parse_featurelist(client, command, actor, args):
    """Parse FEATURELIST and update the Host object."""
    # Strip off ":are supported by this server"
    args = args.rsplit(":", 1)[0]
    # Strip off the nick; we know it's addressed to us.
    _, _, args = args.partition(' ')

    items = args.split()
    for item in items:
        feature, _, value = item.partition("=")

        # Convert integer values to actual integers for convenience
        try:
            value = int(value)
        except (ValueError, TypeError):
            pass

        client.server.features[feature] = value