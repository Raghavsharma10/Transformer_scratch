def render(self, request, redirect_url=None, **kwargs):
        """
        Uses `self.template` to render a response.

        :param request: The current request object.
        :param redirect_url: If given this will return the \
        redirect method instead of rendering the normal template. \
        Renders providing this argument are referred to as a \
        'render redirect' in this documentation.
        :param kwargs: The current context keyword arguments.
        """
        if redirect_url:
            # Redirection is used when we click on `Save` for ordering
            # items on `ListView`. `kwargs` contains `message` but that
            # one is not passing through redirection. That's the reason for using
            # directly `messages` and get message on result template
            if kwargs.get('obj') is None:
                messages.success(request, kwargs.get('message'))
            return self.redirect(request, redirect_url, **kwargs)

        kwargs = self.update_kwargs(request, **kwargs)
        return render(request, self.template, kwargs)