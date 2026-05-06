def ver(self, revision):
        """Clone and change the version."""

        c = self.clone()

        c.version = self._parse_version(self.version)

        return c