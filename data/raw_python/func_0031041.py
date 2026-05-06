def _request(self, url, *,
                 method='GET', headers=None, data=None, result_callback=None):
        """Perform synchronous request.

        :param str url: request URL.
        :param str method: request method.
        :param object data: JSON-encodable object.
        :param object -> object result_callback: result callback.

        :rtype: dict
        :raise: APIError
        """

        retries_left = self._conn_retries

        while True:
            s = self._make_session()
            try:
                cert = None
                if self._client_cert and self._client_key:
                    cert = (self._client_cert, self._client_key)
                elif self._client_cert:
                    cert = self._client_cert

                if self._verify_cert:
                    verify = True
                    if self._ca_certs:
                        verify = self._ca_certs
                else:
                    verify = False

                auth = None
                if self._username and self._password:
                    auth = (self._username, self._password)

                response = s.request(method, url, data=data,
                                     timeout=self._connect_timeout,
                                     cert=cert,
                                     headers=headers,
                                     verify=verify,
                                     auth=auth)
                """:type: requests.models.Response
                """
                if 400 <= response.status_code < 500:
                    raise ClientError(
                        response.status_code, response.content)
                elif response.status_code >= 500:
                    raise ServerError(
                        response.status_code, response.content)

                try:
                    if result_callback:
                        return result_callback(response.content)
                except (ValueError, TypeError) as err:
                    raise MalformedResponse(err) from None

                return response.content

            except (requests.exceptions.RequestException,
                    requests.exceptions.BaseHTTPError) as exc:
                if self._conn_retries is None or retries_left <= 0:
                    raise CommunicationError(exc) from None
                else:
                    retries_left -= 1
                    retry_in = (self._conn_retries - retries_left) * 2
                    self._log.warning('Server communication error: %s. '
                                      'Retrying in %s seconds.', exc, retry_in)
                    time.sleep(retry_in)
                    continue
            finally:
                s.close()