def _authenticate_user(self, user, password):
        """
        Returns the response of authenticating with the given
        user and password.
        """
        headers = self._get_headers()
        params = {
                'username': user,
                'password': password,
                'grant_type': 'password',
                }
        uri = self.uri + '/oauth/token'

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))
        logging.debug("BODY=" + str(params))

        response = requests.post(uri, headers=headers, params=params)
        if response.status_code == 200:
            logging.debug("RESPONSE=" + str(response.json()))
            return response.json()
        else:
            logging.warning("Failed to authenticate %s" % (user))
            response.raise_for_status()