def get_url_kwargs(self, request_kwargs=None, **kwargs):
        """
        Get the kwargs needed to reverse this url.

        :param request_kwargs: The kwargs from the current request. \
        These keyword arguments are only retained if they are present \
        in this bundle's known url_parameters.
        :param kwargs: Keyword arguments that will always be kept.
        """

        if not request_kwargs:
            request_kwargs = getattr(self, 'kwargs', {})

        for k in self.bundle.url_params:
            if k in request_kwargs and not k in kwargs:
                kwargs[k] = request_kwargs[k]
        return kwargs