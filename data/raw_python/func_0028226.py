def connect_websocket(self, **kwargs) -> 'WebSocketContextManager':
        '''
        Creates a WebSocket connection.

        .. warning::

          This method only works with
          :class:`~ai.backend.client.session.AsyncSession`.
        '''
        assert isinstance(self.session, AsyncSession), \
               'Cannot use websockets with sessions in the synchronous mode'
        assert self.method == 'GET', 'Invalid websocket method'
        self.date = datetime.now(tzutc())
        self.headers['Date'] = self.date.isoformat()
        # websocket is always a "binary" stream.
        self.content_type = 'application/octet-stream'
        full_url = self._build_url()
        self._sign(full_url.relative())
        ws_ctx = self.session.aiohttp_session.ws_connect(
            str(full_url),
            autoping=True, heartbeat=30.0,
            headers=self.headers)
        return WebSocketContextManager(self.session, ws_ctx, **kwargs)