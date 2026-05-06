def session_info(self, session_id=None):
        """Get information on session.

        If session_id is None, the default, then return information about this
        session.  If a session ID is given, then get information about that
        session.

        Arguments:
        session_id -- Id of session to get info for, if not this session.

        Return:
        Dictionary of session information.

        """
        if not session_id:
            if not self.started():
                return []
            session_id = self._sid
        status, data = self._rest.get_request('sessions', session_id)
        return data