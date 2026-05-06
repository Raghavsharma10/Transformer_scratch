def request(self, method, endpoint, body=None, timeout=-1):
        """
        Perform a request with a given body to a given endpoint in UpCloud's API.

        Handles errors with __error_middleware.
        """
        if method not in set(['GET', 'POST', 'PUT', 'DELETE']):
            raise Exception('Invalid/Forbidden HTTP method')

        url = '/' + self.api_v + endpoint
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        if body:
            json_body_or_None = json.dumps(body)
        else:
            json_body_or_None = None

        call_timeout = timeout if timeout != -1 else self.timeout

        APIcall = getattr(requests, method.lower())
        res = APIcall('https://api.upcloud.com' + url,
                      data=json_body_or_None,
                      headers=headers,
                      timeout=call_timeout)

        if res.text:
            res_json = res.json()
        else:
            res_json = {}

        return self.__error_middleware(res, res_json)