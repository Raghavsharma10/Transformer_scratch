def start(self):
        """
            Starts the session.

            Starting the session will actually get the API key of the current user
        """

        if NURESTSession.session_stack:
            bambou_logger.critical("Starting a session inside a with statement is not supported.")
            raise Exception("Starting a session inside a with statement is not supported.")

        NURESTSession.current_session = self

        self._authenticate()
        return self