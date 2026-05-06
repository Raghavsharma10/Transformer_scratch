def new_session(self, server=None, session_name=None, user_name=None,
                    existing_session=None):
        """Create a new session or attach to existing.

        Normally, this function is called automatically, and gets its parameter
        values from the environment.  It is provided as a public function for
        cases when extra control over session creation is required in an
        automation script that is adapted to use ReST.

        WARNING:  This function is not part of the original StcPython.py and if
        called directly by an automation script, then that script will not be
        able to revert to using the non-ReST API until the call to this
        function is removed.

        Arguments:
        server           -- STC server (Lab Server) address.  If not set get
                            value from STC_SERVER_ADDRESS environment variable.
        session_name     -- Name part of session ID.  If not set get value from
                            STC_SESSION_NAME environment variable.
        user_name        -- User portion of session ID.  If not set get name of
                            user this script is running as.
        existing_session -- Behavior when session already exists.  Recognized
                            values are 'kill' and 'join'.  If not set get value
                            from EXISTING_SESSION environment variable.  If not
                            set to recognized value, raise exception if session
                            already exists.

        See also: stchttp.StcHttp(), stchttp.new_session()

        Return:
        The internal StcHttp object that is used for this session.  This allows
        the caller to perform additional interactions with the STC ReST API
        beyond what the adapter provides.

        """
        if not server:
            server = os.environ.get('STC_SERVER_ADDRESS')
            if not server:
                raise EnvironmentError('STC_SERVER_ADDRESS not set')
        self._stc = stchttp.StcHttp(server)
        if not session_name:
            session_name = os.environ.get('STC_SESSION_NAME')
            if not session_name or session_name == '__NEW_TEST_SESSION__':
                session_name = None
        if not user_name:
            try:
                # Try to get the name of the current user.
                user_name = getpass.getuser()
            except:
                pass

        if not existing_session:
            # Try to get existing_session from environ if not passed in.
            existing_session = os.environ.get('EXISTING_SESSION')

        if existing_session:
            existing_session = existing_session.lower()
            if existing_session == 'kill':
                # Kill any existing session and create a new one.
                self._stc.new_session(user_name, session_name, True)
                return self._stc
            if existing_session == 'join':
                # Create a new session, or join if already exists.
                try:
                    self._stc.new_session(user_name, session_name, False)
                except RuntimeError as e:
                    if str(e).find('already exists') >= 0:
                        sid = ' - '.join((session_name, user_name))
                        self._stc.join_session(sid)
                    else:
                        raise
                return self._stc

        # Create a new session, raise exception if session already exists.
        self._stc.new_session(user_name, session_name, False)
        return self._stc