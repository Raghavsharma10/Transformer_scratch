def create_session(self, session_id, register=True, session_factory=None):
        """ Creates new session object and returns it.

        @param session_id: Session id. If not provided, will generate a
                           new session id.

        @param register: Should be the session registered in a storage.
                         Websockets don't need it.

        @param session_factory: Use the given (class, args, kwargs) tuple to
                                create the session. Class should derive from
                                `BaseSession`. Normally not needed.

        """
        if session_factory is not None:
            # use custom class to create session
            sess_factory, sess_args, sess_kwargs = session_factory
            s = sess_factory(*sess_args, **sess_kwargs)
        else:
            # use default session and arguments if not using a custom session
            # factory
            s = session.Session(self._connection, self, session_id,
                                self.settings.get('disconnect_delay'))

        if register:
            self._sessions.add(s)

        return s