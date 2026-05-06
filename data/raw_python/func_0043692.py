def output(self, resource):
        """Wrap a resource (as a flask view function).

        This is for cases where the resource does not directly return
        a response object. Now everything should be a Response object.

        :param resource: The resource as a flask view function
        """
        @wraps(resource)
        def wrapper(*args, **kwargs):
            rv = resource(*args, **kwargs)
            rv = self.responder(rv)
            return rv

        return wrapper