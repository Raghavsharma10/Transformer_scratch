def _parse_migrations(self):
        """Build a :class:`Migration` instance."""
        migration = self.parsed['migration']
        options = self._parse_options(migration)
        versions = self._parse_versions(migration, options)
        return Migration(versions, options)