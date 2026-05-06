def _validate_response(self, method, response):
        ''' Helper method to validate the given to a Wunderlist API request is as expected '''
        # TODO Fill this out using the error codes here: https://developer.wunderlist.com/documentation/concepts/formats
        # The expected results can change based on API version, so validate this here
        if self.api_version:
            if response.status_code >= 400:
                raise ValueError('{} {}'.format(response.status_code, str(response.json())))
            if method == 'GET':
                assert response.status_code == 200
            elif method == 'POST':
                assert response.status_code == 201
            elif method == 'PATCH':
                assert response.status_code == 200
            elif method == 'DELETE':
                assert response.status_code == 204