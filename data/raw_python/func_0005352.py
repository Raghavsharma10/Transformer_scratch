def _process(self, resource=None, data={}):
        """Processes the current transaction

        Sends an HTTP request to the PAYDUNYA API server
        """
        # use object's data if no data is passed
        _data = data or self._data
        rsc_url = self.get_rsc_endpoint(resource)
        if _data:
            req = requests.post(rsc_url, data=json.dumps(_data),
                                headers=self.headers)
        else:
            req = requests.get(rsc_url, params=_data,
                               headers=self.headers)
        if req.status_code == 200:
            self._response = json.loads(req.text)
            if int(self._response['response_code']) == 00:
                return (True, self._response)
            else:
                return (False, self._response['response_text'])
        else:
            return (500, "Request Failed")