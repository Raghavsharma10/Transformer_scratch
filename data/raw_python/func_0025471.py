def bind(self, context) -> None:
        """Bind a context to this computation.

        The context allows the computation to convert object specifiers to actual objects.
        """

        # make a computation context based on the enclosing context.
        self.__computation_context = ComputationContext(self, context)

        # re-bind is not valid. be careful to set the computation after the data item is already in document.
        for variable in self.variables:
            assert variable.bound_item is None
        for result in self.results:
            assert result.bound_item is None

        # bind the variables
        for variable in self.variables:
            self.__bind_variable(variable)

        # bind the results
        for result in self.results:
            self.__bind_result(result)