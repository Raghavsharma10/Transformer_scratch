def _regenerate_secret_key(self):
        """Regenerate secret key

        http://www.mediafire.com/developers/core_api/1.3/getting_started/#call_signature
        """
        # Don't regenerate the key if we have none
        if self._session and 'secret_key' in self._session:
            self._session['secret_key'] = (
                int(self._session['secret_key']) * 16807) % 2147483647