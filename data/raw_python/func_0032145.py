def get_schema(self, alias):
        """ Actually resolve the schema for the given alias/URI. """
        if isinstance(alias, dict):
            return alias
        uri = self.get_uri(alias)
        if uri is None:
            raise GraphException('No such schema: %r' % alias)
        uri, schema = self.resolver.resolve(uri)
        return schema