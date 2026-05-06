def fetch(self, **kwargs) -> 'FetchContextManager':
        '''
        Sends the request to the server and reads the response.

        You may use this method either with plain synchronous Session or
        AsyncSession.  Both the followings patterns are valid:

        .. code-block:: python3

          from ai.backend.client.request import Request
          from ai.backend.client.session import Session

          with Session() as sess:
            rqst = Request(sess, 'GET', ...)
            with rqst.fetch() as resp:
              print(resp.text())

        .. code-block:: python3

          from ai.backend.client.request import Request
          from ai.backend.client.session import AsyncSession

          async with AsyncSession() as sess:
            rqst = Request(sess, 'GET', ...)
            async with rqst.fetch() as resp:
              print(await resp.text())
        '''
        assert self.method in self._allowed_methods, \
               'Disallowed HTTP method: {}'.format(self.method)
        self.date = datetime.now(tzutc())
        self.headers['Date'] = self.date.isoformat()
        if self.content_type is not None:
            self.headers['Content-Type'] = self.content_type
        full_url = self._build_url()
        self._sign(full_url.relative())
        rqst_ctx = self.session.aiohttp_session.request(
            self.method,
            str(full_url),
            data=self._pack_content(),
            timeout=_default_request_timeout,
            headers=self.headers)
        return FetchContextManager(self.session, rqst_ctx, **kwargs)