def reflect_static_member(cls, name):
        """Reflect 'name' using ONLY static reflection.

        You most likely want to use ScopeStack.reflect instead.

        Returns:
            Type of 'name', or protocol.AnyType.
        """
        for scope in reversed(cls.scopes):
            try:
                return structured.reflect_static_member(scope, name)
            except (NotImplementedError, KeyError, AttributeError):
                continue

        return protocol.AnyType