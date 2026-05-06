def __continue_session(self):
        """
        Check if the time since the last HTTP request is under the
        session timeout limit. If it's been too long since the last request
        attempt to authenticate again.
        """
        now = time.time()
        diff = abs(now - self.last_request_time)
        timeout_sec = self.session_timeout * 60  # convert minutes to seconds

        if diff >= timeout_sec:
            self.__log('Session timed out, attempting to authenticate')
            self.authenticate()