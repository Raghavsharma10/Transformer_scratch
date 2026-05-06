def modify(self, management_address=None, username=None, password=None,
               connection_type=None):
        """
        Modifies a remote system for remote replication.

        :param management_address: same as the one in `create` method.
        :param username: username for accessing the remote system.
        :param password: password for accessing the remote system.
        :param connection_type: same as the one in `create` method.
        """
        req_body = self._cli.make_body(
            managementAddress=management_address, username=username,
            password=password, connectionType=connection_type)

        resp = self.action('modify', **req_body)
        resp.raise_if_err()
        return resp