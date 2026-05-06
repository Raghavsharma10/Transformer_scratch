def create(self, request, *args, **kwargs):
        """HACK: couldn't get POST to the list endpoint without
        messing up POST for the other list_routes so I'm doing this.
        Maybe something to do with the router?
        """
        return self.list(request, *args, **kwargs)