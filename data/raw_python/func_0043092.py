def header(self):
        """Generates HTTP header for this cookie."""

        if self._send_cookie:
            morsel = Morsel()
            cookie_value = Session.encode_sid(self._config.secret, self._sid)
            if self._config.encrypt_key:
                cipher = AESCipher(self._config.encrypt_key)
                cookie_value = cipher.encrypt(cookie_value)

                if sys.version_info > (3, 0):
                    cookie_value = cookie_value.decode()

            morsel.set(self._config.cookie_name, cookie_value, cookie_value)

            # domain
            if self._config.domain:
                morsel["domain"] = self._config.domain

            # path
            if self._config.path:
                morsel["path"] = self._config.path
            # expires
            if self._expire_cookie:
                morsel["expires"] = "Wed, 01-Jan-1970 00:00:00 GMT"
            elif self._backend_client.expires:
                morsel["expires"] = self._backend_client.expires.strftime(EXPIRES_FORMAT)

            # secure
            if self._config.secure:
                morsel["secure"] = True

            # http only
            if self._config.http_only:
                morsel["httponly"] = True

            return morsel.OutputString()
        else:
            return None