def _request(self, method, url, params=None, headers=None, data=None):
        """Common handler for all the HTTP requests."""
        if not params:
            params = {}

        # set default headers
        if not headers:
            headers = {
                'accept': '*/*'
            }
            if method == 'POST' or method == 'PUT':
                headers.update({'Content-Type': 'application/json'})
        try:
            response = requests.request(method=method, url=self.host + self.key + url, params=params,
                                        headers=headers, data=data)

            try:
                response.raise_for_status()

                response_code = response.status_code
                success = True if response_code // 100 == 2 else False
                if response.text:
                    try:
                        data = response.json()
                    except ValueError:
                        data = response.content
                else:
                    data = ''

                response_headers = response.headers

                return BurpResponse(success=success, response_code=response_code, data=data,
                                    response_headers=response_headers)
            except ValueError as e:
                return BurpResponse(success=False, message="JSON response could not be decoded {}.".format(e))
            except requests.exceptions.HTTPError as e:
                if response.status_code == 400:
                    return BurpResponse(success=False, response_code=400, message='Bad Request')
                else:
                    return BurpResponse(
                        message='There was an error while handling the request. {}'.format(response.content),
                        success=False)
        except Exception as e:
            return BurpResponse(success=False, message='Eerror is %s' % e)