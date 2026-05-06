def dispatch(self, request, *args, **kwargs):
        """
        Overrides the custom dispatch method to raise a Http404
        if the current user does not have view permissions.
        """
        self.request = request
        self.args = args
        self.kwargs = kwargs

        if not self.can_view(request.user):
            raise http.Http404

        return super(CMSView, self).dispatch(request, *args, **kwargs)