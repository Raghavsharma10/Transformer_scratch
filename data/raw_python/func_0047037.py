def authentication_required(function):
        """Annotation for methods that require auth."""
        def wrapped(self, *args, **kwargs):
            if not (self.token or self.apiKey):
                msg = "You must be authenticated to use this method"
                raise AuthenticationError(msg)
            else:
                return function(self, *args, **kwargs)
        return wrapped