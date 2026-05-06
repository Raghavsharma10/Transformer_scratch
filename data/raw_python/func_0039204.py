def required(self, fn):
        """Request decorator. Forces authentication."""
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            if (not self._check_auth()
               # Don't try to force authentication if the request is part
               # of the authentication process - otherwise we end up in a
               # loop.
               and request.blueprint != self.blueprint.name):
                return redirect(url_for("%s.login" % self.blueprint.name,
                                        next=request.url))
            return fn(*args, **kwargs)
        return decorated