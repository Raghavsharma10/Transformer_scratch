def register_function(self, function, name=None):
        """
        Register function to be called from EPC client.

        :type  function: callable
        :arg   function: Function to publish.
        :type      name: str
        :arg       name: Name by which function is published.

        This method returns the given `function` as-is, so that you
        can use it as a decorator.

        """
        if name is None:
            name = function.__name__
        self.funcs[name] = function
        return function