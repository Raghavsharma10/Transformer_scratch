def stream_pty(self) -> 'StreamPty':
        '''
        Opens a pseudo-terminal of the kernel (if supported) streamed via
        websockets.

        :returns: a :class:`StreamPty` object.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        request = Request(self.session,
                          'GET', '/stream/kernel/{}/pty'.format(self.kernel_id),
                          params=params)
        return request.connect_websocket(response_cls=StreamPty)