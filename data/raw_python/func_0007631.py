def ensure_connected(f):
        """
        Check that the connection is valid both before and
        after the function is invoked.
        """
        @functools.wraps(f)
        def wrapper(*args):
            self = args[0]
            self.invalid()
            try:
                return f(*args)
            finally:
                self.invalid()
        return wrapper