def _post(self, uri, data):
        """
        Simple POST request for a given path.
        """
        headers = self._get_headers()

        logging.debug("URI=" + str(uri))
        logging.debug("BODY=" + json.dumps(data))

        response = self.session.post(uri, headers=headers,
                data=json.dumps(data))
        if response.status_code in [200, 204]:
            try:
                return response.json()
            except ValueError:
                return "{}"
        else:
            logging.error(response.content)
            response.raise_for_status()