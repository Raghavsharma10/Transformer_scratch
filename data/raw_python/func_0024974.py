def _delete(self, uri):
        """
        Simple DELETE operation for a given path.
        """
        headers = self._get_headers()

        response = self.session.delete(uri, headers=headers)

        # Will return a 204 on successful delete
        if response.status_code == 204:
            return response
        else:
            logging.error(response.content)
            response.raise_for_status()