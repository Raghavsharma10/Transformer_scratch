def _login(self):
        """Login to Asterisk Manager Interface."""
        self._sendAction("login", (
            ("Username", self._amiuser),
            ("Secret", self._amipass),
            ("Events", "off"),
        ))
        resp = self._getResponse()
        if resp.get("Response") == "Success":
            return True
        else:
            raise Exception("Authentication to Asterisk Manager Interface Failed.")