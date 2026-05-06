def expect_exact(self, *args, **kwargs):
        """This does not attempt to duplicate the expect_exact API,
        but just sets self.before to the latest response line."""
        response = self._recvline()
        self.before = response.strip()