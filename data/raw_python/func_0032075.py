def connectRoute(amp, router, receiver, protocol):
    """
    Connect the given receiver to a new box receiver for the given
    protocol.

    After connecting this router to an AMP server, use this method
    similarly to how you would use C{reactor.connectTCP} to establish a new
    connection to an HTTP, SMTP, or IRC server.

    @param receiver: An L{IBoxReceiver} which will be started when a route
        to a receiver for the given protocol is found.

    @param protocol: The name of a protocol which the AMP peer to which
        this router is connected has an L{IBoxReceiverFactory}.

    @return: A L{Deferred} which fires with C{receiver} when the route is
        established.
    """
    route = router.bindRoute(receiver)
    d = amp.callRemote(
        Connect,
        origin=route.localRouteName,
        protocol=protocol)
    def cbGotRoute(result):
        route.connectTo(result['route'])
        return receiver
    d.addCallback(cbGotRoute)
    return d