def discover(self, topic):
        '''Run the discovery mechanism'''
        logger.info('Discovering on topic %s', topic)
        producers = []
        for lookupd in self._lookupd:
            logger.info('Discovering on %s', lookupd)
            try:
                # Find all the current producers on this instance
                for producer in lookupd.lookup(topic)['producers']:
                    logger.info('Found producer %s on %s', producer, lookupd)
                    producers.append(
                        (producer['broadcast_address'], producer['tcp_port']))
            except ClientException:
                logger.exception('Failed to query %s', lookupd)

        new = []
        for host, port in producers:
            conn = self._connections.get((host, port))
            if not conn:
                logger.info('Discovered %s:%s', host, port)
                new.append(self.connect(host, port))
            elif not conn.alive():
                logger.info('Reconnecting to %s:%s', host, port)
                if conn.connect():
                    conn.setblocking(0)
                    self.reconnected(conn)
            else:
                logger.debug('Connection to %s:%s still alive', host, port)

        # And return all the new connections
        return [conn for conn in new if conn]