def send_method_request(self, method: str, method_params: dict) -> dict:
        """
        Sends user-defined method and method params
        """
        url = '/'.join((self.METHOD_URL, method))
        method_params['v'] = self.API_VERSION
        if self._access_token:
            method_params['access_token'] = self._access_token
        response = self.post(url, method_params, timeout=10)
        response.raise_for_status()
        return json.loads(response.text)