def _get_depencency_var_name(self, dependency):
        """
        Returns the variable name assigned to the given dependency or None if the dependency has
        not yet been registered.

        Args:
            dependency (str): Thet dependency that needs to be imported.

        Returns:
            str or None
        """
        for dep_path, var_name in self.dependencies:
            if dep_path == dependency:
                return var_name