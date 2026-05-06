def parse_refresh_header(self, refresh):
        """
        >>> parse_refresh_header("1; url=http://example.com/")
        (1.0, 'http://example.com/')
        >>> parse_refresh_header("1; url='http://example.com/'")
        (1.0, 'http://example.com/')
        >>> parse_refresh_header("1")
        (1.0, None)
        >>> parse_refresh_header("blah")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: invalid literal for float(): blah
        """
        ii = refresh.find(";")
        if ii != -1:
            pause, newurl_spec = float(refresh[:ii]), refresh[ii+1:]
            jj = newurl_spec.find("=")
            key = None
            if jj != -1:
                key, newurl = newurl_spec[:jj], newurl_spec[jj+1:]
                newurl = self.clean_refresh_url(newurl)
            if key is None or key.strip().lower() != "url":
                raise ValueError()
        else:
            pause, newurl = float(refresh), None
        return pause, newurl