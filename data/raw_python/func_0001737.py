def get_full_url(self, url):
        # type: (str) -> str
        """Get full url including any additional parameters

        Args:
            url (str): URL for which to get full url

        Returns:
            str: Full url including any additional parameters
        """
        request = Request('GET', url)
        preparedrequest = self.session.prepare_request(request)
        return preparedrequest.url