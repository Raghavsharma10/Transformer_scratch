def resolve(self, name):
        """Call IStructured.resolve across all scopes and return first hit."""
        for scope in reversed(self.scopes):
            try:
                return structured.resolve(scope, name)
            except (KeyError, AttributeError):
                continue

        raise AttributeError(name)