def add_function(self, function):
        """
        Adds the function to the list of registered functions.
        """
        function = self.build_function(function)
        if function.name in self.functions:
            raise FunctionAlreadyRegistered(function.name)
        self.functions[function.name] = function