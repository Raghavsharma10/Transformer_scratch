def get_vhost_permissions(self, vname):
        """
        :returns: list of dicts, or an empty list if there are no permissions.

        :param string vname: Name of the vhost to set perms on.
        """
        vname = quote(vname, '')
        path = Client.urls['vhost_permissions_get'] % (vname,)
        conns = self._call(path, 'GET')
        return conns