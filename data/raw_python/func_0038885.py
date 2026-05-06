def url(self):
        """The URL as a string of the resource."""
        if not self._url[2].endswith('/'):
            self._url[2] += '/'
        return RestURL.url.__get__(self)