def _restart_session(self, session):
        """Restarts and re-establishes session

        :param session: The session to restart
        """
        # remove old session key, if socket is None, that means the
        # session was closed by user and there is no need to restart.
        if session.socket is not None:
            self.log.info("Attempting restart session for Monitor Id %s."
                          % session.monitor_id)
            del self.sessions[session.socket.fileno()]
            session.stop()
            session.start()
            self.sessions[session.socket.fileno()] = session