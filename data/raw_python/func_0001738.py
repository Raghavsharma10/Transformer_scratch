def get_url_for_get(url, parameters=None):
        # type: (str, Optional[Dict]) -> str
        """Get full url for GET request including parameters

        Args:
            url (str): URL to download
            parameters (Optional[Dict]): Parameters to pass. Defaults to None.

        Returns:
            str: Full url

        """
        spliturl = urlsplit(url)
        getparams = OrderedDict(parse_qsl(spliturl.query))
        if parameters is not None:
            getparams.update(parameters)
        spliturl = spliturl._replace(query=urlencode(getparams))
        return urlunsplit(spliturl)