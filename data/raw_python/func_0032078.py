def accept(self, origin, protocol):
        """
        Create a new route attached to a L{IBoxReceiver} created by the
        L{IBoxReceiverFactory} with the indicated protocol.

        @type origin: C{unicode}
        @param origin: The identifier of a route on the peer which will be
            associated with this connection.  Boxes sent back by the protocol
            which is created in this call will be sent back to this route.

        @type protocol: C{unicode}
        @param protocol: The name of the protocol to which to establish a
            connection.

        @raise ProtocolUnknown: If no factory can be found for the named
            protocol.

        @return: A newly created C{unicode} route identifier for this
            connection (as the value of a C{dict} with a C{'route'} key).
        """
        for factory in self.store.powerupsFor(IBoxReceiverFactory):
            # XXX What if there's a duplicate somewhere?
            if factory.protocol == protocol:
                receiver = factory.getBoxReceiver()
                route = self.router.bindRoute(receiver)
                # This might be better implemented using a hook on the box.
                # See Twisted ticket #3479.
                self.reactor.callLater(0, route.connectTo, origin)
                return {'route': route.localRouteName}
        raise ProtocolUnknown()