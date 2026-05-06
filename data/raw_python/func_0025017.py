def _post(self, uri, data, headers=None):
        """
        Simple POST request for a given uri path.
        """
        if not headers:
            headers = self._get_headers()

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))
        logging.debug("BODY=" + str(data))

        response = self.session.post(uri, headers=headers,
                data=json.dumps(data))

        logging.debug("STATUS=" + str(response.status_code))
        if response.status_code in [200, 201]:
            return response.json()
        else:
            logging.error(b"ERROR=" + response.content)
            response.raise_for_status()