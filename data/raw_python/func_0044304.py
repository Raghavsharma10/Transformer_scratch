def _url_for_queue(self, conn, name):
        """
        Given a queue name, return the URL for it.

        :param conn: SQS API connection
        :type conn: :py:class:`botocore:SQS.Client`
        :param name: queue name, or None for all queues in config.
        :type name: str
        :return: queue URL
        :rtype: str
        """
        res = conn.get_queue_url(QueueName=name)
        return res['QueueUrl']