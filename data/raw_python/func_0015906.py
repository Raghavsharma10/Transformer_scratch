def delete_exchange(self, vhost, name):
        """
        Delete the named exchange from the named vhost. The API returns a 204
        on success, in which case this method returns True, otherwise the
        error is raised.

        :param string vhost: Vhost where target exchange was created
        :param string name: The name of the exchange to delete.
        :returns bool: True on success.
        """
        vhost = quote(vhost, '')
        name = quote(name, '')
        path = Client.urls['exchange_by_name'] % (vhost, name)
        self._call(path, 'DELETE')
        return True