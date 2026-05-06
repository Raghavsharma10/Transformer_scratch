def mark_dead(self, connection, now=None):
        """
        Mark the connection as dead (failed). Remove it from the live pool and
        put it on a timeout.

        :arg connection: the failed instance
        """
        # allow inject for testing purposes
        now = now if now else time.time()
        try:
            self.connections.remove(connection)
        except ValueError:
            # connection not alive or another thread marked it already, ignore
            return
        else:
            dead_count = self.dead_count.get(connection, 0) + 1
            self.dead_count[connection] = dead_count
            timeout = self.dead_timeout * 2 ** min(dead_count - 1,
                                                   self.timeout_cutoff)
            self.dead.put((now + timeout, connection))
            logger.warning(
                'Connection %r has failed for %i times in a row,'
                ' putting on %i second timeout.',
                connection, dead_count, timeout
            )