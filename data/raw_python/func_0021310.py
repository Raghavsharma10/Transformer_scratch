def remove_writer(self, address):
        """ Remove a writer address from the routing table, if present.
        """
        log_debug("[#0000]  C: <ROUTING> Removing writer %r", address)
        self.routing_table.writers.discard(address)
        log_debug("[#0000]  C: <ROUTING> table=%r", self.routing_table)