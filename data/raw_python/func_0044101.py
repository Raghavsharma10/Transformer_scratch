def get_pager(self, *path, **kwargs):
        """ A generator for all the results a resource can provide. The pages
        are lazily loaded. """
        page_arg = kwargs.pop('page_size', None)
        limit_arg = kwargs.pop('limit', None)
        kwargs['limit'] = page_arg or limit_arg or self.default_page_size
        return self.adapter.get_pager(self.get, path, kwargs)