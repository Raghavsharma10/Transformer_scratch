def queue_callback(self, session, block_id, data):
        """
        Queues up a callback event to occur for a session with the given
        payload data.  Will block if the queue is full.

        :param session: the session with a defined callback function to call.
        :param block_id: the block_id of the message received.
        :param data: the data payload of the message received.
        """
        self._queue.put((session, block_id, data))