def _put(self, uri, data):
        """
        Simple PUT operation for a given path.
        """
        headers = self._get_headers()

        logging.debug("URI=" + str(uri))
        logging.debug("BODY=" + json.dumps(data))

        response = self.session.put(uri, headers=headers,
                data=json.dumps(data))
        if response.status_code in [201, 204]:
            return data
        else:
            logging.error(response.content)
            response.raise_for_status()