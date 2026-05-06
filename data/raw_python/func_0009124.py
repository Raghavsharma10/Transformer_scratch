def ack(self, tup_id):
        """Called when a bolt acknowledges a Tuple in the topology.

        :param tup_id: the ID of the Tuple that has been fully acknowledged in
                       the topology.
        :type tup_id: str
        """
        self.failed_tuples.pop(tup_id, None)
        try:
            del self.unacked_tuples[tup_id]
        except KeyError:
            self.logger.error("Received ack for unknown tuple ID: %r", tup_id)