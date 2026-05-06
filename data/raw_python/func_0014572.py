def new_session(self, user_name=None, session_name=None,
                    kill_existing=False, analytics=None):
        """Create a new test session.

        The test session is identified by the specified user_name and optional
        session_name parameters.  If a session name is not specified, then the
        server will create one.

        Arguments:
        user_name     -- User name part of session ID.
        session_name  -- Session name part of session ID.
        kill_existing -- If there is an existing session, with the same session
                         name and user name, then terminate it before creating
                         a new session
        analytics     -- Optional boolean value to disable or enable analytics
                         for new session.  None will use setting configured on
                         server.

        Return:
        True is session started, False if session was already started.

        """
        if self.started():
            return False
        if not session_name or not session_name.strip():
            session_name = ''
        if not user_name or not user_name.strip():
            user_name = ''
        params = {'userid': user_name, 'sessionname': session_name}
        if analytics not in (None, ''):
            params['analytics'] = str(analytics).lower()
        try:
            status, data = self._rest.post_request('sessions', None, params)
        except resthttp.RestHttpError as e:
            if kill_existing and str(e).find('already exists') >= 0:
                self.end_session('kill', ' - '.join((session_name, user_name)))
            else:
                raise RuntimeError('failed to create session: ' + str(e))

            # Starting session
            if self._dbg_print:
                print('===> starting session')
            status, data = self._rest.post_request('sessions', None, params)
            if self._dbg_print:
                print('===> OK, started')

        sid = data['session_id']
        if self._dbg_print:
            print('===> session ID:', sid)
            print('===> URL:', self._rest.make_url('sessions', sid))

        self._rest.add_header('X-STC-API-Session', sid)
        self._sid = sid
        return sid