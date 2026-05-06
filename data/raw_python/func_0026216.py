def _post_login_page(self, login_url):
        """Login to HydroQuebec website."""
        data = {"login": self.username,
                "_58_password": self.password}

        try:
            raw_res = yield from self._session.post(login_url,
                                                    data=data,
                                                    timeout=self._timeout,
                                                    allow_redirects=False)
        except OSError:
            raise PyHydroQuebecError("Can not submit login form")
        if raw_res.status != 302:
            raise PyHydroQuebecError("Login error: Bad HTTP status code. "
                                     "Please check your username/password.")
        return True