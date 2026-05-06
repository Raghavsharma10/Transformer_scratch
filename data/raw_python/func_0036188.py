def upload_submit(self, upload_request):
        """The method is submitting dataset upload"""

        path = '/api/1.0/upload/save'
        return self._api_post(definition.DatasetUploadResponse, path, upload_request)