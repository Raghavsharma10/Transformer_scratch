def getmembers_static(cls):
        """Gets members (vars) from all scopes using ONLY static information.

        You most likely want to use ScopeStack.getmembers instead.

        Returns:
            Set of available vars.

        Raises:
            NotImplementedError if any scope fails to implement 'getmembers'.
        """
        names = set()
        for scope in cls.scopes:
            names.update(structured.getmembers_static(scope))

        return names