def _request(self, url, *,
                 method='GET', headers=None, data=None, result_callback=None):
        """Perform asynchronous request.

        :param str url: request URL.
        :param str method: request method.
        :param dict headers: request headers.
        :param object data: JSON-encodable object.
        :param object -> object result_callback: result callback.

        :rtype: dict
        :raise: APIError
        """

        request = self._prepare_request(url, method, headers, data)

        retries_left = self._conn_retries

        while True:
            try:
                response = yield self._client.fetch(request)
                try:
                    if result_callback:
                        return result_callback(response.body)
                except (ValueError, TypeError) as err:
                    raise MalformedResponse(err) from None

                return response.body

            except httpclient.HTTPError as err:
                resp_body = err.response.body \
                    if err.response is not None else None
                if err.code == 599:
                    if self._conn_retries is None or retries_left <= 0:
                        raise CommunicationError(err) from None
                    else:
                        retries_left -= 1
                        retry_in = (self._conn_retries - retries_left) * 2
                        self._log.warning('Server communication error: %s. '
                                          'Retrying in %s seconds.', err,
                                          retry_in)
                        yield gen.sleep(retry_in)
                        continue
                elif 400 <= err.code < 500:
                    raise ClientError(err.code, resp_body) from None

                raise ServerError(err.code, resp_body) from None