def getmembers_runtime(self):
        """Gets members (vars) from all scopes using ONLY runtime information.

        You most likely want to use ScopeStack.getmembers instead.

        Returns:
            Set of available vars.

        Raises:
            NotImplementedError if any scope fails to implement 'getmembers'.
        """
        names = set()
        for scope in self.scopes:
            names.update(structured.getmembers_runtime(scope))

        return names