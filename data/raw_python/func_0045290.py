def flattened_params_url(self, path_prefix, *paths, **kparams):
        """ url builder helper to make /api/now/v1/table paths for GET requests. Snow is Woe."""

        base = self.base_url + path_prefix
        for p in paths:
            base += "/"
            base += p
        if kparams:
            base += "?"
            # use %r in val?
            base += '&'.join("%s=%s" % (key,val) for (key,val) in kparams.items())
        return base