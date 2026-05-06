def unbind(self):
        """Unlisten and close each bound item."""
        for variable in self.variables:
            self.__unbind_variable(variable)
        for result in self.results:
            self.__unbind_result(result)