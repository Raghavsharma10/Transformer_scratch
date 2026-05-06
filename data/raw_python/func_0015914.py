def delete_queue(self, vhost, qname):
        """
        Deletes the named queue from the named vhost.

        :param string vhost: Vhost housing the queue to be deleted.
        :param string qname: Name of the queue to delete.

        Note that if you just want to delete the messages from a queue, you
        should use purge_queue instead of deleting/recreating a queue.
        """
        vhost = quote(vhost, '')
        qname = quote(qname, '')
        path = Client.urls['queues_by_name'] % (vhost, qname)
        return self._call(path, 'DELETE', headers=Client.json_headers)