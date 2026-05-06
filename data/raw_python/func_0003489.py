def register(self, f, *args, **kwargs):
        """
        Register a function and arguments to be called later.
        """
        self._functions.append(lambda: f(*args, **kwargs))