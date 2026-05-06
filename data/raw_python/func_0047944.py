def _function(self, type, name, args=""):
        """
        Returns a context manager for writing a function.

        :param str type: The return type of the function
        :param str name: The name of the functino
        :param str args: The argument specification for the function
        """
        return FunctionManager(self, type=type, name=name, args=args)