def post(self, path, data):
        """
        Generic POST with headers
        """
        uri = self.config.get_target() + path
        headers = self._post_headers()

        logging.debug("URI=POST " + str(uri))
        logging.debug("HEADERS=" + str(headers))
        logging.debug("BODY=" + str(data))

        response = self.session.post(uri, headers=headers,
                data=json.dumps(data))
        if response.status_code in (200, 201, 202):
            return response.json()
        elif response.status_code == 401:
            raise predix.admin.cf.config.CloudFoundryLoginError('token invalid')
        else:
            logging.debug("STATUS=" + str(response.status_code))
            logging.debug("CONTENT=" + str(response.content))
            response.raise_for_status()