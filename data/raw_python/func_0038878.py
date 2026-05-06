def url(self):
        """The URL as a string of the resource."""
        urlparts = self._url
        if self.__post__:
            urlparts = list(urlparts)
            urlparts[3] = '' # Clear out query string on POST
            if self.__token__ is not None: # But not the token
                urlparts[3] = compat.urlencode({'token': self.__token__})
        return compat.urlunsplit(urlparts)