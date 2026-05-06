def process_exception(self, request, exception):
        """
        Return a redirect response for the :class:`~fluent_contents.extensions.HttpRedirectRequest`
        """
        if isinstance(exception, HttpRedirectRequest):
            return HttpResponseRedirect(exception.url, status=exception.status)
        else:
            return None