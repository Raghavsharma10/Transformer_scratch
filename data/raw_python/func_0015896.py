def delete_vhost(self, vname):
        """
        Deletes a vhost from the server. Note that this also deletes any
        exchanges or queues that belong to this vhost.

        :param string vname: Name of the vhost to delete from the server.
        """
        vname = quote(vname, '')
        path = Client.urls['vhosts_by_name'] % vname
        return self._call(path, 'DELETE')