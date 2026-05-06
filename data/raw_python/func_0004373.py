def get(self, uri):
        """
            Sends a GET request.

            @param uri: Uri of Service API.
            @param data: Requesting Data. Default: None

            @raise NetworkAPIClientError: Client failed to access the API.
        """
        request = None

        try:

            request = requests.get(
                self._url(uri),
                auth=self._auth_basic(),
                headers=self._header()
            )

            request.raise_for_status()

            try:
                return request.json()
            except Exception:
                return request

        except HTTPError:
            try:
                error = request.json()
                self.logger.error(error)
                err = error.get('detail', '')
            except:
                err = request
            raise NetworkAPIClientError(err)
        finally:
            self.logger.info('URI: %s', uri)
            if request:
                self.logger.info('Status Code: %s',
                                 request.status_code if request else '')
                self.logger.info('X-Request-Id: %s',
                                 request.headers.get('x-request-id'))
                self.logger.info('X-Request-Context: %s',
                                 request.headers.get('x-request-context'))