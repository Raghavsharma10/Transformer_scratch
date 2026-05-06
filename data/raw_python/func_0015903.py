def get_exchange(self, vhost, name):
        """
        Gets a single exchange which requires a vhost and name.

        :param string vhost: The vhost containing the target exchange
        :param string name: The name of the exchange
        :returns: dict

        """
        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['exchange_by_name'] % (vhost, name)
        exch = self._call(path, 'GET')
        return exch