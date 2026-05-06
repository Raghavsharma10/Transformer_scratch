def register(self, function):
        """Register a function in the function registry.
        The function will be automatically instantiated if not already an
        instance.
        """
        function = inspect.isclass(function) and function() or function
        name = function.name
        self[name] = function