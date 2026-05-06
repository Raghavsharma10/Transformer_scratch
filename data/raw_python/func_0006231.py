def create_session(self, server_host, server_port, honeypot_id):
        """
            Creates a new session.

        :param server_host: IP address of the server
        :param server_port: Server port
        :return: A new `BaitSession` object.
        """
        protocol = self.__class__.__name__.lower()
        session = BaitSession(protocol, server_host, server_port, honeypot_id)
        self.sessions[session.id] = session
        return session