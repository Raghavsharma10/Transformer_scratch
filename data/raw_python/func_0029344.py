def wrap(self, func):
        """ Wrap :func: to perform aggregation on :func: call.

        Should be called with view instance methods.
        """
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return self.aggregate()
            except KeyError:
                return func(*args, **kwargs)
        return wrapper