def _radius_auth(self, server, username, password):
        """
        Authenticate the given username/password against the RADIUS server
        described by `server`.
        """
        client = self._get_client(server)
        packet = self._get_auth_packet(username, password, client)
        return self._perform_radius_auth(client, packet)