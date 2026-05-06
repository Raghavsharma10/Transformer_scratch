def get(self, request, *args, **kwargs):
        """Override default get function to use token if there is one to retrieve object. If a
        subclass should use their own GET implementation, token_from_kwargs should be called if
        that detail view should be accessible via token."""

        self.object = self.get_object()

        allow_anonymous = kwargs.get("allow_anonymous", False)

        # We only want to redirect is that setting is true, and we're not allowing anonymous users
        if self.redirect_correct_path and not allow_anonymous:

            # Also we obviously only want to redirect if the URL is wrong
            if self.request.path != self.object.get_absolute_url():
                return HttpResponsePermanentRedirect(self.object.get_absolute_url())

        context = self.get_context_data(object=self.object)
        response = self.render_to_response(context)

        # If we have an unpublished article....
        if self.object.published is None or self.object.published > timezone.now():

            # And the user doesn't have permission to view this
            if not request.user.is_staff and not allow_anonymous:
                response = redirect_unpublished_to_login_or_404(
                    request=request,
                    next_url=self.object.get_absolute_url(),
                    next_params=request.GET)

            # Never cache unpublished articles
            add_never_cache_headers(response)
        else:
            response["Vary"] = "Accept-Encoding"

        return response