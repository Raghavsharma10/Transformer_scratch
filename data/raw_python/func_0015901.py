def delete_permission(self, vname, username):
        """
        Delete permission for a given username on a given vhost. Both
        must already exist.

        :param string vname: Name of the vhost to set perms on.
        :param string username: User to set permissions for.
        """
        vname = quote(vname, '')
        path = Client.urls['vhost_permissions'] % (vname, username)
        return self._call(path, 'DELETE')