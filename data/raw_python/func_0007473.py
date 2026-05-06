def _display_token(self):
        """
        Display token information or redirect to login prompt if none is
        available.
        """
        if self.token is None:
            return "301 Moved", "", {"Location": "/login"}

        return ("200 OK",
                self.TOKEN_TEMPLATE.format(
                    access_token=self.token["access_token"]),
                {"Content-Type": "text/html"})