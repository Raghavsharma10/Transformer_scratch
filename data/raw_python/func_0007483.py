def header(self, name, default=None):
        """
        Returns the value of the HTTP header identified by `name`.
        """
        wsgi_header = "HTTP_{0}".format(name.upper())

        try:
            return self.env_raw[wsgi_header]
        except KeyError:
            return default