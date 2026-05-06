def dispatch(self):
        """Handles dispatching of the request."""
        method_name = 'on_' + self.environ['REQUEST_METHOD'].lower()
        method = getattr(self, method_name, None)
        if method:
            return method()
        else:
            return self.on_bad_method()