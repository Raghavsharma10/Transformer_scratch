def upload_status(self, upload_id):
        """The method is checking status of uploaded dataset"""

        path = '/api/1.0/upload/status'
        query = 'id={}'.format(upload_id)
        return self._api_get(definition.DatasetUploadStatusResponse, path, query)