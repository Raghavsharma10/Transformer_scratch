def distribute_ready(self):
        '''Distribute the ready state across all of the connections'''
        connections = [c for c in self.connections() if c.alive()]
        if len(connections) > self._max_in_flight:
            raise NotImplementedError(
                'Max in flight must be greater than number of connections')
        else:
            # Distribute the ready count evenly among the connections
            for count, conn in distribute(self._max_in_flight, connections):
                # We cannot exceed the maximum RDY count for a connection
                if count > conn.max_rdy_count:
                    logger.info(
                        'Using max_rdy_count (%i) instead of %i for %s RDY',
                        conn.max_rdy_count, count, conn)
                    count = conn.max_rdy_count
                logger.info('Sending RDY %i to %s', count, conn)
                conn.rdy(count)