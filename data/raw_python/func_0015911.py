def purge_queues(self, queues):
        """
        Purge all messages from one or more queues.

        :param list queues: A list of ('qname', 'vhost') tuples.
        :returns: True on success

        """
        for name, vhost in queues:
            vhost = quote(vhost, '')
            name = quote(name, '')
            path = Client.urls['purge_queue'] % (vhost, name)
            self._call(path, 'DELETE')
        return True