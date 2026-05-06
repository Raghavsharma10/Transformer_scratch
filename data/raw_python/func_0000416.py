def _request(self, url, request_type='GET', **params):
        """Send a request to the Minut Point API."""
        try:
            _LOGGER.debug('Request %s %s', url, params)
            response = self.request(
                request_type, url, timeout=TIMEOUT.seconds, **params)
            response.raise_for_status()
            _LOGGER.debug('Response %s %s %.200s', response.status_code,
                          response.headers['content-type'], response.json())
            response = response.json()
            if 'error' in response:
                raise OSError(response['error'])
            return response
        except OSError as error:
            _LOGGER.warning('Failed request: %s', error)