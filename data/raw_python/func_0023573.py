def session(self, value):
        """Set session token

        value -- dict returned by user/get_session_token"""

        # unset session token
        if value is None:
            self._session = None
            return

        if not isinstance(value, dict):
            raise ValueError("session info is required")

        session_parsed = {}

        for key in ["session_token", "time", "secret_key"]:
            if key not in value:
                raise ValueError("Missing parameter: {}".format(key))
            session_parsed[key] = value[key]

        for key in ["ekey", "pkey"]:
            # nice to have, but not mandatory
            if key in value:
                session_parsed[key] = value[key]

        self._session = session_parsed