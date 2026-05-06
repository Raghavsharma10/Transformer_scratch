def _get(self, uri, params=None, headers=None):
        """
        Simple GET request for a given path.
        """
        if not headers:
            headers = self._get_headers()

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))

        response = self.session.get(uri, headers=headers, params=params)
        logging.debug("STATUS=" + str(response.status_code))
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(b"ERROR=" + response.content)
            response.raise_for_status()