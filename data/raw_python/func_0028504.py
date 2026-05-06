def _get_content(self, params, mode_translate):
        """
            This method gets the token and makes the header variable that 
            will be used in connection authentication. After that, calls 
            the _make_request() method to return the desired data.
        """
        token = self._get_token()
        headers = {'Authorization': 'Bearer '+ token}
        parameters = params
        translation_url = mode_translate
        return self._make_request(parameters, translation_url, headers)