def _default_client(jws_client, reactor, key, alg):
    """
    Make a client if we didn't get one.
    """
    if jws_client is None:
        pool = HTTPConnectionPool(reactor)
        agent = Agent(reactor, pool=pool)
        jws_client = JWSClient(HTTPClient(agent=agent), key, alg)
    return jws_client