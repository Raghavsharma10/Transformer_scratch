def get_queue_depths(self, vhost, names=None):
        """
        Get the number of messages currently sitting in either the queue
        names listed in 'names', or all queues in 'vhost' if no 'names' are
        given.

        :param str vhost: Vhost where queues in 'names' live.
        :param list names: OPTIONAL - Specific queues to show depths for. If
                None, show depths for all queues in 'vhost'.
        """

        vhost = quote(vhost, '')
        if not names:
            # get all queues in vhost
            path = Client.urls['queues_by_vhost'] % vhost
            queues = self._call(path, 'GET')
            for queue in queues:
                depth = queue['messages']
                print("\t%s: %s" % (queue, depth))
        else:
            # get the named queues only.
            for name in names:
                depth = self.get_queue_depth(vhost, name)
                print("\t%s: %s" % (name, depth))