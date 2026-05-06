def get_url(self, url):
        """
        Internally used to retrieve the contents of a URL
        """
        _r = self.br.open(url)

        # check that we've not been redirected to the login page
        if self.br.geturl().startswith(self.AUTH_URL):
            raise AuthRequiredException
        elif self.br.geturl().startswith(self.ERROR_URL):
            raise RequestErrorException
        else:
            return _r.read()