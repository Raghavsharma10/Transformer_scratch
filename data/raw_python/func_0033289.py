def _update_session_expiration(self):
        """
        Updates a redis item to expire later since it has been interacted with
        recently
        """

        session_time = oz.settings["session_time"]

        if session_time:
            self.redis().expire(self._session_key, session_time)