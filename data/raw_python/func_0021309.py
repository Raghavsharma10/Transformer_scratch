def deactivate(self, address):
        """ Deactivate an address from the connection pool,
        if present, remove from the routing table and also closing
        all idle connections to that address.
        """
        log_debug("[#0000]  C: <ROUTING> Deactivating address %r", address)
        # We use `discard` instead of `remove` here since the former
        # will not fail if the address has already been removed.
        self.routing_table.routers.discard(address)
        self.routing_table.readers.discard(address)
        self.routing_table.writers.discard(address)
        log_debug("[#0000]  C: <ROUTING> table=%r", self.routing_table)
        super(RoutingConnectionPool, self).deactivate(address)