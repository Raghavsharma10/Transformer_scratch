def delete_user(self, id):
        """
        Delete user with given id.
        """
        self.assert_has_permission('scim.write')

        uri = self.uri + '/Users/%s' % id
        headers = self._get_headers()

        logging.debug("URI=" + str(uri))
        logging.debug("HEADERS=" + str(headers))

        response = self.session.delete(uri, headers=headers)
        logging.debug("STATUS=" + str(response.status_code))
        if response.status_code == 200:
            return response
        else:
            logging.error(response.content)
            response.raise_for_status()