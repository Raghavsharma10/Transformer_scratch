def _api_request(self, action, data, request_type, headers=None):
        """
        Make Request to Sailthru API with given data and api key, format and signature hash
        """
        if 'file' in data:
            file_data = {'file': open(data['file'], 'rb')}
        else:
            file_data = None

        return self._http_request(action, self._prepare_json_payload(data), request_type, file_data, headers)