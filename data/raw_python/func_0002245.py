def _get_installations(self):
        """ Get information about installations """
        response = None
        for base_url in urls.BASE_URLS:
            urls.BASE_URL = base_url
            try:
                response = requests.get(
                    urls.get_installations(self._username),
                    headers={
                        'Cookie': 'vid={}'.format(self._vid),
                        'Accept': 'application/json,'
                                  'text/javascript, */*; q=0.01',
                    })
                if 2 == response.status_code // 100:
                    break
                elif 503 == response.status_code:
                    continue
                else:
                    raise ResponseError(response.status_code, response.text)
            except requests.exceptions.RequestException as ex:
                raise RequestError(ex)

        _validate_response(response)
        self.installations = json.loads(response.text)