def _man_args(self, f):
        """
        Returns number of mandatory arguments required by given function.
        """
        argcount = f.func_code.co_argcount

        # account for "self" getting passed to class instance methods
        if isinstance(f, types.MethodType):
            argcount -= 1

        if f.func_defaults is None:
            return argcount

        return argcount - len(f.func_defaults)