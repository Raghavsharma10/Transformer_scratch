def get_queues(self, vhost=None):
        """
        Get all queues, or all queues in a vhost if vhost is not None.
        Returns a list.

        :param string vhost: The virtual host to list queues for. If This is
                    None (the default), all queues for the broker instance
                    are returned.
        :returns: A list of dicts, each representing a queue.
        :rtype: list of dicts

        """
        if vhost:
            vhost = quote(vhost, '')
            path = Client.urls['queues_by_vhost'] % vhost
        else:
            path = Client.urls['all_queues']

        queues = self._call(path, 'GET')
        return queues or list()