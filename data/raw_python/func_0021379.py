def _create_session(self, test_connection=False):
        """
        Create a consulate.session object, and query for its leader to ensure
        that the connection is made.

        :param test_connection: call .leader() to ensure that the connection
            is valid
        :type test_connection: bool
        :return consulate.Session instance
        """
        session = consulate.Session(host=self.host, port=self.port)
        if test_connection:
            session.status.leader()
        return session