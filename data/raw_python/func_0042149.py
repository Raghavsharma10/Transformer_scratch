def get_url_kwargs(self, request_kwargs=None, **kwargs):
        """
        If request_kwargs is not specified, self.kwargs is used instead.

        If 'object' is one of the kwargs passed. Replaces it with
        the value of 'self.slug_field' on the given object.
        """

        if not request_kwargs:
            request_kwargs = getattr(self, 'kwargs', {})

        kwargs = super(ModelCMSView, self).get_url_kwargs(request_kwargs,
                                                          **kwargs)
        obj = kwargs.pop('object', None)
        if obj:
            kwargs[self.slug_url_kwarg] = getattr(obj, self.slug_field, None)
        elif self.slug_url_kwarg in request_kwargs:
            kwargs[self.slug_url_kwarg] = request_kwargs[self.slug_url_kwarg]

        return kwargs