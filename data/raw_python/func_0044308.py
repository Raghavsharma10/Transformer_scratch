def show_queue(self, name=None, count=10, delete=False):
        """
        Show up to ``count`` messages from the queue named ``name``. If ``name``
        is None, show for each queue in our config. If ``delete`` is True,
        delete the messages after showing them.

        :param name: queue name, or None for all queues in config.
        :type name: str
        :param count: maximum number of messages to get from queue
        :type count: int
        :param delete: whether or not to delete messages after receipt
        :type delete: bool
        """
        logger.debug('Connecting to SQS API')
        conn = client('sqs')
        if name is not None:
            queues = [name]
        else:
            queues = self._all_queue_names
        for q_name in queues:
            try:
                self._show_one_queue(conn, q_name, count, delete=delete)
            except Exception:
                logger.error("Error showing queue '%s'", q_name, exc_info=1)