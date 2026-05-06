def start_session(self):
        """Start the underlying APIs sessions

        Calling this is not required, it will be called automatically if
        a method that needs a session is called

        @return bool
        """
        self._android_api.start_session()
        self._manga_api.cr_start_session()
        return self.session_started