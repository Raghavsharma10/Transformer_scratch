def get_queue_depth(self, vhost, name):
        """
        Get the number of messages currently in a queue. This is a convenience
         function that just calls :meth:`Client.get_queue` and pulls
         out/returns the 'messages' field from the dictionary it returns.

        :param string vhost: The vhost of the queue being queried.
        :param string name: The name of the queue to query.
        :returns: Number of messages in the queue
        :rtype: integer

        """
        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['queues_by_name'] % (vhost, name)
        queue = self._call(path, 'GET')
        depth = queue['messages']

        return depth