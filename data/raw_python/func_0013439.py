def setUrl(self, url):
        """
        Attempt to safely set the URL by string.
        """
        if isUrl(url):
            self._url = url
        else:
            raise exceptions.BadUrlException(url)
        return self