def _patch(self, uri, data):
        """
        Simple PATCH operation for a given path.

        The body is expected to list operations to perform to update
        the data.  Operations include:
            - add
            - remove
            - replace
            - move
            - copy
            - test

        [
             { "op": "test", "path": "/a/b/c", "value": "foo" },
        ]
        """
        headers = self._get_headers()
        response = self.session.patch(uri, headers=headers,
                data=json.dumps(data))

        # Will return a 204 on successful patch
        if response.status_code == 204:
            return response
        else:
            logging.error(response.content)
            response.raise_for_status()