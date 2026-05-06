def delete(self, dataset):
        """The method is deleting dataset by it's id"""

        url = self._get_url('/api/1.0/meta/dataset/{}/delete'.format(dataset))

        json_data = ''
        binary_data = json_data.encode()

        headers = self._get_request_headers()
        req = urllib.request.Request(url, binary_data, headers)
        resp = urllib.request.urlopen(req)   
        str_response = resp.read().decode('utf-8')
        if str_response != '"successful"' or resp.status < 200 or resp.status >= 300:
            msg = 'Dataset has not been deleted, because of the following error(s): {}'.format(str_response)
            raise ValueError(msg)