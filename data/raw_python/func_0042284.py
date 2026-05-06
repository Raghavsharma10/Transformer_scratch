def get_different_page(self, request, page):
        """
        Returns a url that preserves the current querystring
        while changing the page requested to `page`.
        """

        if page:
            qs = request.GET.copy()
            qs['page'] = page
            return "%s?%s" % (request.path_info, qs.urlencode())
        return None