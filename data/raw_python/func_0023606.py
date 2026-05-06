def check_connections(self):
        '''Connect to all the appropriate instances'''
        logger.info('Checking connections')
        if self._lookupd:
            self.discover(self._topic)

        # Make sure we're connected to all the prescribed hosts
        for hostspec in self._nsqd_tcp_addresses:
            logger.debug('Checking nsqd instance %s', hostspec)
            host, port = hostspec.split(':')
            port = int(port)
            conn = self._connections.get((host, port), None)
            # If there is no connection to it, we have to try to connect
            if not conn:
                logger.info('Connecting to %s:%s', host, port)
                self.connect(host, port)
            elif not conn.alive():
                # If we've connected to it before, but it's no longer alive,
                # we'll have to make a decision about when to try to reconnect
                # to it, if we need to reconnect to it at all
                if conn.ready_to_reconnect():
                    logger.info('Reconnecting to %s:%s', host, port)
                    if conn.connect():
                        conn.setblocking(0)
                        self.reconnected(conn)
            else:
                logger.debug('Checking freshness')
                now = time.time()
                time_check = math.ceil(now - self.last_recv_timestamp)
                if time_check >= ((self.heartbeat_interval * 2) / 1000.0):
                    if conn.ready_to_reconnect():
                        logger.info('Reconnecting to %s:%s', host, port)
                        if conn.connect():
                            conn.setblocking(0)
                            self.reconnected(conn)