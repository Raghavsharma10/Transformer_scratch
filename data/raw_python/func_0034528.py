def negotiate_safe(self, name, params):
        """
        `name` and `params` are sent in the HTTP request by the client. Check
        if the extension name is supported by this extension, and validate the
        parameters. Returns a dict with accepted parameters, or None if not
        accepted.
        """
        for param in params.iterkeys():
            if param not in self.defaults:
                return

        try:
            return dict(self.negotiate(name, params))
        except (KeyError, ValueError, AssertionError):
            pass