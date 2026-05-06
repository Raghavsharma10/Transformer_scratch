def _clean_dead_sessions(self):
        """
        Traverses sessions to determine if any sockets
        were removed (indicates a stopped session).
        In these cases, remove the session.
        """
        for sck in list(self.sessions.keys()):
            session = self.sessions[sck]
            if session.socket is None:
                del self.sessions[sck]