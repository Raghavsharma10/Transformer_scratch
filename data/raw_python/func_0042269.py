def get_string_from_view(self, request, view_name, url_kwargs,
                                                render_type='string'):

        """
        Returns a string that is a rendering of the view given a
        request, view_name, and the original url_kwargs. Makes the
        following changes the view before rendering:

        * Sets can_submit to False.
        * Adds action_url to the context. This is the url where \
        this view actually lives.
        * Sets the default base_template to be 'cms/partial.html'

        This will always call GET and never POST as any actions
        that modify data should take place on the original
        url and not like this.

        :param request: The request object.
        :param view_name: The name of the view that you want.
        :param url_kwargs: The url keyword arguments that came \
        with the request object. The view itself is responsible \
        to remove arguments that would not be part of a normal match \
        for that view. This is done by calling  the `get_url_kwargs` \
        method on the view.
        :param render_type: The render type to use. Defaults to \
        'string'.
        """

        response = ""
        try:
            view, name = self.get_initialized_view_and_name(view_name,
                                    render_type=render_type,
                                    can_submit=False,
                                    base_template='cms/partial.html',
                                    request=request, kwargs=url_kwargs)

            if isinstance(view, URLAlias):
                view_name = view.get_view_name(view_name)
                bundle = view.get_bundle(self, url_kwargs, {})
                if bundle and isinstance(bundle, Bundle):
                    return bundle.get_string_from_view(request, view_name,
                                                    url_kwargs,
                                                    render_type=render_type)

            elif view:
                if view and name and view.can_view(request.user):
                    response = self._render_view_as_string(view, name, request,
                                                           url_kwargs)
        except http.Http404:
            pass
        return response