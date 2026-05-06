def get_exchanges(self, vhost=None):
        """
        :returns: A list of dicts
        :param string vhost: A vhost to query for exchanges, or None (default),
            which triggers a query for all exchanges in all vhosts.

        """
        if vhost:
            vhost = quote(vhost, '')
            path = Client.urls['exchanges_by_vhost'] % vhost
        else:
            path = Client.urls['all_exchanges']

        exchanges = self._call(path, 'GET')
        return exchanges