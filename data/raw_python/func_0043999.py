def _get_callargs(self, *args, **kwargs):
        """
        Retrieve all arguments that `self.func` needs and
        return a dictionary with call arguments.
        """
        callargs = getcallargs(self.func, *args, **kwargs)
        return callargs