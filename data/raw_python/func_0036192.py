def verify(self, dataset, publication_date, source, refernce_url):
        """The method is verifying dataset by it's id"""

        path = '/api/1.0/meta/verifydataset'
        req = definition.DatasetVerifyRequest(dataset, publication_date, source, refernce_url)
        result = self._api_post(definition.DatasetVerifyResponse, path, req)
        if result.status == 'failed':
            ver_err = '\r\n'.join(result.errors)
            msg = 'Dataset has not been verified, because of the following error(s): {}'.format(ver_err)
            raise ValueError(msg)