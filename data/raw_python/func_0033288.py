def _session_key(self):
        """Gets the redis key for a session"""

        if not hasattr(self, "_cached_session_key"):
            session_id_bytes = self.get_secure_cookie("session_id")
            session_id = None

            if session_id_bytes:
                try:
                    session_id = session_id_bytes.decode('utf-8')
                except:
                    pass

            if not session_id:
                session_id = oz.redis_sessions.random_hex(20)

            session_time = oz.settings["session_time"]
            kwargs = dict(
                name="session_id",
                value=session_id.encode('utf-8'),
                domain=oz.settings.get("cookie_domain"),
                httponly=True,
            )
            if session_time:
                kwargs["expires_days"] = round(session_time/60/60/24)

            self.set_secure_cookie(**kwargs)

            password_salt = oz.settings["session_salt"]
            self._cached_session_key = "session:%s:v4" % oz.redis_sessions.password_hash(session_id, password_salt=password_salt)

        return self._cached_session_key