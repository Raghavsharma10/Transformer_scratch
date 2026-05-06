def _add_dependency(self, dependency, var_name=None):
        """
        Adds the given dependency and returns the variable name to use to access it. If `var_name`
        is not given then a random one will be created.

        Args:
            dependency (str):
            var_name (str, optional):

        Returns:
            str
        """
        if var_name is None:
            var_name = next(self.temp_var_names)
        # Don't add duplicate dependencies
        if (dependency, var_name) not in self.dependencies:
            self.dependencies.append((dependency, var_name))
        return var_name