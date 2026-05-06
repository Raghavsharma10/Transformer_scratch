def is_installed(self):
        """Return True if the bundle is installed."""

        r = self.library.resolve(self.identity.vid)

        return r is not None