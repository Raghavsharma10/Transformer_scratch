def post_url(self, url, form):
        """
        Internally used to retrieve the contents of a URL using
        the POST request method.
        The `form` parameter is a mechanize.HTMLForm object
        This method will use a POST request type regardless of the method
        used in the `form`.
        """
        _r = self.br.open(url, form.click_request_data()[1])

        # check that we've not been redirected to the login page or an error occured
        if self.br.geturl().startswith(self.AUTH_URL):
            raise AuthRequiredException
        elif self.br.geturl().startswith(self.ERROR_URL):
            raise RequestErrorException
        else:
            return _r.read()