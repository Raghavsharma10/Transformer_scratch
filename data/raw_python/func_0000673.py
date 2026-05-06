def _session_check(self):
        """Attempt to authenticate the user through a session file.

        This process is done to avoid having to authenticate the user every
        single time. It uses a session file that is saved when a valid session
        is captured and then reused. Because sessions can expire, we need to
        test the session prior to calling the user authenticated. Right now
        that is done with a test string found in an unauthenticated session.
        This approach is not an ideal method, but it works.
        """
        if not os.path.exists(SESSION_FILE):
            self._log.debug("Session file does not exist")
            return False
        with open(SESSION_FILE, 'rb') as f:
            cookies = requests.utils.cookiejar_from_dict(pickle.load(f))
            self._session.cookies = cookies
            self._log.debug("Loaded cookies from session file")
        response = self._session.get(url=self.TEST_URL, headers=self.HEADERS)
        if self.TEST_KEY in str(response.content):
            self._log.debug("Session file appears invalid")
            return False
        self._is_authenticated = True
        self._process_state()
        return True