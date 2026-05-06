def get(self, path):
        """
        Generic GET with headers
        """
        uri = self.config.get_target() + path
        headers = self._get_headers()

        logging.debug("URI=GET " + str(uri))
        logging.debug("HEADERS=" + str(headers))

        response = self.session.get(uri, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise predix.admin.cf.config.CloudFoundryLoginError('token invalid')
        else:
            response.raise_for_status()