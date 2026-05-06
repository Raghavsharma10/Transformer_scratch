def verify(self, connection_type=None):
        """
        Verifies and update the remote system settings.

        :param connection_type: same as the one in `create` method.
        """
        req_body = self._cli.make_body(connectionType=connection_type)

        resp = self.action('verify', **req_body)
        resp.raise_if_err()
        return resp