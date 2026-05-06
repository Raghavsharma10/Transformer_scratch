def read(self):
        '''Read from any of the connections that need it'''
        # We'll check all living connections
        connections = [c for c in self.connections() if c.alive()]

        if not connections:
            # If there are no connections, obviously we return no messages, but
            # we should wait the duration of the timeout
            time.sleep(self._timeout)
            return []

        # Not all connections need to be written to, so we'll only concern
        # ourselves with those that require writes
        writes = [c for c in connections if c.pending()]
        try:
            readable, writable, exceptable = select.select(
                connections, writes, connections, self._timeout)
        except exceptions.ConnectionClosedException:
            logger.exception('Tried selecting on closed client')
            return []
        except select.error:
            logger.exception('Error running select')
            return []

        # If we returned because the timeout interval passed, log it and return
        if not (readable or writable or exceptable):
            logger.debug('Timed out...')
            return []

        responses = []
        # For each readable socket, we'll try to read some responses
        for conn in readable:
            try:
                for res in conn.read():
                    # We'll capture heartbeats and respond to them automatically
                    if (isinstance(res, Response) and res.data == HEARTBEAT):
                        logger.info('Sending heartbeat to %s', conn)
                        conn.nop()
                        logger.debug('Setting last_recv_timestamp')
                        self.last_recv_timestamp = time.time()
                        continue
                    elif isinstance(res, Error):
                        nonfatal = (
                            exceptions.FinFailedException,
                            exceptions.ReqFailedException,
                            exceptions.TouchFailedException
                        )
                        if not isinstance(res.exception(), nonfatal):
                            # If it's not any of the non-fatal exceptions, then
                            # we have to close this connection
                            logger.error(
                                'Closing %s: %s', conn, res.exception())
                            self.close_connection(conn)
                    responses.append(res)
                    logger.debug('Setting last_recv_timestamp')
                    self.last_recv_timestamp = time.time()
            except exceptions.NSQException:
                logger.exception('Failed to read from %s', conn)
                self.close_connection(conn)
            except socket.error:
                logger.exception('Failed to read from %s', conn)
                self.close_connection(conn)

        # For each writable socket, flush some data out
        for conn in writable:
            try:
                conn.flush()
            except socket.error:
                logger.exception('Failed to flush %s', conn)
                self.close_connection(conn)

        # For each connection with an exception, try to close it and remove it
        # from our connections
        for conn in exceptable:
            self.close_connection(conn)

        return responses