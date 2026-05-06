def _request_api(self, **kwargs):
        """Wrap the calls the url, with the given arguments.

        :param str url: Url to call with the given arguments
        :param str method: [POST | GET] Method to use on the request
        :param int status: Expected status code
        """
        _url = kwargs.get('url')
        _method = kwargs.get('method', 'GET')
        _status = kwargs.get('status', 200)

        counter = 0
        if _method not in ['GET', 'POST']:
            raise ValueError('Method is not GET or POST')

        while True:
            try:
                res = REQ[_method](_url, cookies=self._cookie)
                if res.status_code == _status:
                    break
                else:
                    raise BadStatusException(res.content)
            except requests.exceptions.BaseHTTPError:
                if counter < self._retries:
                    counter += 1
                    continue
                raise MaxRetryError
        self._last_result = res
        return res