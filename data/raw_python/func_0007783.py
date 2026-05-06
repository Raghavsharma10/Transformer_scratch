def request(self, method, path, **kwargs):
        """Send a :class:`requests.Request` and demand a
        :class:`requests.Response`
        """
        if path:
            url = '%s/%s' % (self.url.rstrip('/'), path.lstrip('/'))
        else:
            url = self.url

        request_params = self._get_request_params(method=method,
                                                  url=url, **kwargs)
        request_params = self.pre_send(request_params)

        sanitized_params = self._sanitize_request_params(request_params)
        start_time = time.time()
        response = super(HTTPServiceClient, self).request(**sanitized_params)

        # Log request and params (without passwords)
        log.debug(
            '%s HTTP [%s] call to "%s" %.2fms',
            response.status_code, method, response.url,
            (time.time() - start_time) * 1000)
        auth = sanitized_params.pop('auth', None)
        log.debug('HTTP request params: %s', sanitized_params)
        if auth:
            log.debug('Authentication via HTTP auth as "%s"', auth[0])

        response.is_ok = response.status_code < 300
        if not self.is_acceptable(response, request_params):
            raise HTTPServiceError(response)
        response = self.post_send(response, **request_params)
        return response