def parse(self):
        """Check input and return a :class:`Migration` instance."""
        if not self.parsed.get('migration'):
            raise ParseError(u"'migration' key is missing", YAML_EXAMPLE)
        self.check_dict_expected_keys(
            {'options', 'versions'}, self.parsed['migration'], 'migration',
        )
        return self._parse_migrations()