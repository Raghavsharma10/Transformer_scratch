def get_queue(self, vhost, name):
        """
        Get a single queue, which requires both vhost and name.

        :param string vhost: The virtual host for the queue being requested.
            If the vhost is '/', note that it will be translated to '%2F' to
            conform to URL encoding requirements.
        :param string name: The name of the queue being requested.
        :returns: A dictionary of queue properties.
        :rtype: dict

        """
        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['queues_by_name'] % (vhost, name)
        queue = self._call(path, 'GET')
        return queue