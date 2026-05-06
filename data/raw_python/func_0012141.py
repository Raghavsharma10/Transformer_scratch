def select_peer(peer_addrs, service, routing_id, method):
    '''Choose a target from the available peers for a singular message

    :param peer_addrs:
        the ``(host, port)``s of the peers eligible to handle the RPC, and
        possibly a ``None`` entry if this hub can handle it locally
    :type peer_addrs: list
    :param service: the service of the message
    :type service: anything hash-able
    :param routing_id: the routing_id of the message
    :type routing_id: int
    :param method: the message method name
    :type method: string

    :returns: one of the provided peer_addrs

    There is no reason to call this method directly, but it may be useful to
    override it in a Hub subclass.

    This default implementation uses ``None`` if it is available (prefer local
    handling), then falls back to a random selection.
    '''
    if any(p is None for p in peer_addrs):
        return None
    return random.choice(peer_addrs)