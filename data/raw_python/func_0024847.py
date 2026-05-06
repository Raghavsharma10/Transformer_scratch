def delete(self, path, data=None, params=None):
        """
        Generic DELETE with headers
        """
        uri = self.config.get_target() + path
        headers = {
            'Authorization': self.config.get_access_token()
            }

        logging.debug("URI=DELETE " + str(uri))
        logging.debug("HEADERS=" + str(headers))

        response = self.session.delete(
            uri, headers=headers, params=params, data=json.dumps(data))

        if response.status_code == 204:
            return response
        else:
            logging.debug("STATUS=" + str(response.status_code))
            logging.debug("CONTENT=" + str(response.content))
            response.raise_for_status()