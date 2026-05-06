def purge_queue(self, vhost, name):
        """
        Purge all messages from a single queue. This is a convenience method
        so you aren't forced to supply a list containing a single tuple to
        the purge_queues method.

        :param string vhost: The vhost of the queue being purged.
        :param string name: The name of the queue being purged.
        :rtype: None

        """
        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['purge_queue'] % (vhost, name)
        return self._call(path, 'DELETE')