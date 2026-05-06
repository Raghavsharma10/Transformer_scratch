def create_vhost(self, vname):
        """
        Creates a vhost on the server to house exchanges.

        :param string vname: The name to give to the vhost on the server
        :returns: boolean
        """
        vname = quote(vname, '')
        path = Client.urls['vhosts_by_name'] % vname
        return self._call(path, 'PUT',
                                 headers=Client.json_headers)