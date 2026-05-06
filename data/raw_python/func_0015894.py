def get_vhost(self, vname):
        """
        Returns the attributes of a single named vhost in a dict.

        :param string vname: Name of the vhost to get.
        :returns dict vhost: Attribute dict for the named vhost

        """

        vname = quote(vname, '')
        path = Client.urls['vhosts_by_name'] % vname
        vhost = self._call(path, 'GET', headers=Client.json_headers)
        return vhost