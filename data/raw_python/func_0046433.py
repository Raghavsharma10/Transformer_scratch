def _parse_created(client, command, actor, args):
    """Parse CREATED and update the Host object."""
    m = re.search("This server was created (.+)$", args)
    if m:
        client.server.created = m.group(1)