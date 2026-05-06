def paginate(self, url, key, params=None):
        """
        Fetch a sequence of paginated resources from the API endpoint.  The
        initial request to ``url`` and all subsequent requests must respond
        with a JSON object; the field specified by ``key`` must be a list,
        whose elements will be yielded, and the next request will be made to
        the URL in the ``.links.pages.next`` field until the responses no
        longer contain that field.

        :param str url: the URL to make the initial request of.  If ``url``
            begins with a forward slash, :attr:`endpoint` is prepended to it;
            otherwise, ``url`` is treated as an absolute URL.
        :param str key: the field on each page containing a list of values to
            yield
        :param dict params: parameters to add to the initial URL's query
            string.  A ``"per_page"`` parameter may be included to override
            the default :attr:`per_page` setting.
        :rtype: generator of decoded JSON values
        :raises ValueError: if a response body is not an object or ``key`` is
            not one of its keys
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if params is None:
            params = {}
        if self.per_page is not None and "per_page" not in params:
            params = dict(params, per_page=self.per_page)
        page = self.request(url, params=params)
        while True:
            try:
                objects = page[key]
            except (KeyError, TypeError):
                raise ValueError('{0!r}: not a key of the response body'\
                                 .format(key))
            for obj in objects:
                yield obj
            try:
                url = page["links"]["pages"]["next"]
            except KeyError:
                break
            page = self.request(url)