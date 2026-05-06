def get_uri(self, alias):
        """ Get the URI for a given alias. A registered URI will return itself,
        otherwise ``None`` is returned. """
        if alias in self.aliases.keys():
            return self.aliases[alias]
        if alias in self.aliases.values():
            return alias
        raise GraphException('No such schema: %r' % alias)