def getmembers(self):
        """Gets members (vars) from all scopes, using both runtime and static.

        This method will attempt both static and runtime getmembers. This is the
        recommended way of getting available members.

        Returns:
            Set of available vars.

        Raises:
            NotImplementedError if any scope fails to implement 'getmembers'.
        """
        names = set()
        for scope in self.scopes:
            if isinstance(scope, type):
                names.update(structured.getmembers_static(scope))
            else:
                names.update(structured.getmembers_runtime(scope))

        return names