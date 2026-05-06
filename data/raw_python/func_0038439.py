def _max_args(self, f):
        """
        Returns maximum number of arguments accepted by given function.
        """
        if f.func_defaults is None:
            return f.func_code.co_argcount

        return f.func_code.co_argcount + len(f.func_defaults)