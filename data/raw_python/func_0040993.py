def _handle_response(response):
        """
        Handle the response and possible failures.
        :param Response response: Response data.
        :return: A dictionary or a string with response data.
        :raises: NeverBounceAPIError if the API call fails.
        """
        if not response.ok:
            raise NeverBounceAPIError(response)
        if response.headers.get('Content-Type') == 'application/octet-stream':
            return response.iter_lines()

        try:
            resp = response.json()
        except ValueError:
             raise InvalidResponseError('Failed to handle the response content-type {}.'.format(
                 response.headers.get('Content-Type'))
             )
        if 'success' in resp and not resp['success']:
            if 'msg' in resp and resp['msg'] == 'Authentication failed':
                raise AccessTokenExpired
            else:
                raise NeverBounceAPIError(response)
        return resp