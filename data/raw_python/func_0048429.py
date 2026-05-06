def _get_url(url):
        """Returns a URL string.

        If the ``url`` parameter is a ParsedResult from `urlparse` the full url
        will be unparsed and made into a string. Otherwise the ``url``
        parameter is returned as is.

        :param url: ``str`` || ``object``
        """
        if isinstance(url, urlparse.ParseResult):
            return urlparse.urlunparse(url)
        else:
            return url