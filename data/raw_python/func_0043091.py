def _load_cookie(self):
        """Loads HTTP Cookie from environ"""

        cookie = SimpleCookie(self._environ.get('HTTP_COOKIE'))
        vishnu_keys = [key for key in cookie.keys() if key == self._config.cookie_name]

        # no session was started yet
        if not vishnu_keys:
            return

        morsel = cookie[vishnu_keys[0]]
        morsel_value = morsel.value

        if self._config.encrypt_key:
            cipher = AESCipher(self._config.encrypt_key)
            morsel_value = cipher.decrypt(morsel_value)

        received_sid = Session.decode_sid(self._config.secret, morsel_value)
        if received_sid:
            self._sid = received_sid
        else:
            logging.warn("found cookie with invalid signature")