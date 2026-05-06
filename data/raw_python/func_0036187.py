def upload_verify(self, file_location, dataset=None):
        """This method is verifiing posted file on server"""

        path = '/api/1.0/upload/verify'
        query = 'doNotGenerateAdvanceReport=true&filePath={}'.format(file_location)
        if dataset:
            query = 'doNotGenerateAdvanceReport=true&filePath={}&datasetId={}'.format(file_location, dataset)

        return self._api_get(definition.UploadVerifyResponse, path, query)