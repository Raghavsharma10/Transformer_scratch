def _normalize_check_url(self, check_url):
        """
        Normalizes check_url by:

        * Adding the `http` scheme if missing
        * Adding or replacing port with `self.port`
        """

        # TODO: Write tests for this method
        split_url = urlsplit(check_url)
        host = splitport(split_url.path or split_url.netloc)[0]
        return '{0}://{1}:{2}'.format(self.scheme, host, self.port)