def _make_full_url(self, url):
        """Given base and relative URL, construct the full URL.

        :param str url: relative URL.

        :return: full URL.
        :rtype: str
        """

        return SLASH.join([self._api_base_url, url.lstrip(SLASH)])