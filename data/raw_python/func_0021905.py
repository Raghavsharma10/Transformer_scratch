def _get_function_name(self, fn, default="None"):
        """ Return name of function, using default value if function not defined
        """
        if fn is None:
            fn_name = default
        else:
            fn_name = fn.__name__
        return fn_name