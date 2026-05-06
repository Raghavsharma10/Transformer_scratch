def required(self, func):
        """
        Decorator function with basic and token authentication handler
        """
        @wraps(func)
        def decorated(*args, **kwargs):
            """
            Actual wrapper to run the auth checks.
            """
            is_valid, user = self.authenticate()
            if not is_valid:
                return self.auth_failed()
            kwargs['user'] = user
            return func(*args, **kwargs)
        return decorated