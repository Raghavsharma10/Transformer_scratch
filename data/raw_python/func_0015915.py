def get_messages(self, vhost, qname, count=1,
                     requeue=False, truncate=None, encoding='auto'):
        """
        Gets <count> messages from the queue.

        :param string vhost: Name of vhost containing the queue
        :param string qname: Name of the queue to consume from
        :param int count: Number of messages to get.
        :param bool requeue: Whether to requeue the message after getting it.
            This will cause the 'redelivered' flag to be set in the message on
            the queue.
        :param int truncate: The length, in bytes, beyond which the server will
            truncate the message before returning it.
        :returns: list of dicts. messages[msg-index]['payload'] will contain
                the message body.
        """

        vhost = quote(vhost, '')
        base_body = {'count': count, 'requeue': requeue, 'encoding': encoding}
        if truncate:
            base_body['truncate'] = truncate
        body = json.dumps(base_body)

        qname = quote(qname, '')
        path = Client.urls['get_from_queue'] % (vhost, qname)
        messages = self._call(path, 'POST', body,
                                     headers=Client.json_headers)
        return messages