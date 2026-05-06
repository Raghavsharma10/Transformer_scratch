def scope(self, key, *tags, default=None):
        """Only apply tags and default for top-level key, effectively scoping the tags."""
        scope = self._scopes[key]
        tags = self._ensure_exclamation(tags)
        default = default if not default or default.startswith("!") else "!" + default

        if scope:
            scope[0] = scope[0] + tags
            scope[1] = default if default else scope[1]
        else:
            scope.append(tags)
            scope.append(default)