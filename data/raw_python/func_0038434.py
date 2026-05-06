def add(self, f, name=None, types=None, required=None):
        """
        Adds a new method to the jsonrpc service.

        Arguments:
        f -- the remote function
        name -- name of the method in the jsonrpc service
        types -- list or dictionary of the types of accepted arguments
        required -- list of required keyword arguments

        If name argument is not given, function's own name will be used.

        Argument types must be a list if positional arguments are used or a
        dictionary if keyword arguments are used in the method in question.

        Argument required MUST be used only for methods requiring keyword
        arguments, not for methods accepting positional arguments.
        """
        if name is None:
            fname = f.__name__  # Register the function using its own name.
        else:
            fname = name

        self.method_data[fname] = {'method': f}

        if types is not None:
            self.method_data[fname]['types'] = types

            if required is not None:
                self.method_data[fname]['required'] = required