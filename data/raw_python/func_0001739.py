def get_url_params_for_post(url, parameters=None):
        # type: (str, Optional[Dict]) -> Tuple[str, Dict]
        """Get full url for POST request and all parameters including any in the url

        Args:
            url (str): URL to download
            parameters (Optional[Dict]): Parameters to pass. Defaults to None.

        Returns:
            Tuple[str, Dict]: (Full url, parameters)

        """
        spliturl = urlsplit(url)
        getparams = OrderedDict(parse_qsl(spliturl.query))
        if parameters is not None:
            getparams.update(parameters)
        spliturl = spliturl._replace(query='')
        full_url = urlunsplit(spliturl)
        return full_url, getparams