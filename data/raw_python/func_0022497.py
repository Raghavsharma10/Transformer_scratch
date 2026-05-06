def reflect_runtime_member(self, name):
        """Reflect 'name' using ONLY runtime reflection.

        You most likely want to use ScopeStack.reflect instead.

        Returns:
            Type of 'name', or protocol.AnyType.
        """
        for scope in reversed(self.scopes):
            try:
                return structured.reflect_runtime_member(scope, name)
            except (NotImplementedError, KeyError, AttributeError):
                continue

        return protocol.AnyType