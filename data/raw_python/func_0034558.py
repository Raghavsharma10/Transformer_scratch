def resurrect(self, force=False):
        """
        Attempt to resurrect a connection from the dead pool. It will try to
        locate one (not all) eligible (it's timeout is over) connection to
        return to th live pool.

        :arg force: resurrect a connection even if there is none eligible (used
            when we have no live connections)

        """
        # no dead connections
        if self.dead.empty():
            return

        try:
            # retrieve a connection to check
            timeout, connection = self.dead.get(block=False)
        except Empty:
            # other thread has been faster and the queue is now empty
            return

        if not force and timeout > time.time():
            # return it back if not eligible and not forced
            self.dead.put((timeout, connection))
            return

        # either we were forced or the connection is elligible to be retried
        self.connections.append(connection)
        logger.info('Resurrecting connection %r (force=%s).', connection, force)