def delete_file(self, id):
        """
        Delete file.

        Remove the specified file
        
          curl -XDELETE 'https://<canvas>/api/v1/files/<file_id>' \
               -H 'Authorization: Bearer <token>'
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("DELETE /api/v1/files/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/files/{id}".format(**path), data=data, params=params, no_data=True)