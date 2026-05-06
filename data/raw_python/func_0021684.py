def _build_request(self, method, url, params=None):
        """Build a function to do an API request

        "We have to go deeper" or "It's functions all the way down!"
        """
        full_params = self._get_base_params()
        if params is not None:
            full_params.update(params)
        try:
            request_func = lambda u, d: \
                getattr(self._connector, method.lower())(u, params=d,
                    headers=self._request_headers)
        except AttributeError:
            raise ApiException('Invalid request method')
        # TODO: need to catch a network here and raise as ApiNetworkException

        def do_request():
            logger.debug('Sending %s request "%s" with params: %r',
                method, url, full_params)
            try:
                resp = request_func(url, full_params)
                logger.debug('Received response code: %d', resp.status_code)
            except requests.RequestException as err:
                raise ApiNetworkException(err)

            try:
                resp_json = resp.json()
            except TypeError:
                resp_json = resp.json

            method_returns_list = False
            try:
                resp_json['error']
            except TypeError:
                logger.warn('Api method did not return map: %s', method)
                method_returns_list = True
            except KeyError:
                logger.warn('Api method did not return map with error key: %s', method)

            if method_returns_list is None:
                raise ApiBadResponseException(resp.content)
            elif method_returns_list:
                data = resp_json
            else:
                try:
                    if resp_json['error']:
                        raise ApiError('%s: %s' % (resp_json['code'], resp_json['message']))
                except KeyError:
                    data = resp_json
                else:
                    data = resp_json['data']
                    self._do_post_request_tasks(data)
            self._last_response = resp
            return data
        return do_request