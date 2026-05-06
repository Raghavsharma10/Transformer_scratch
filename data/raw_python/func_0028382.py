def stream_execute(self, code: str = '', *,
                       mode: str = 'query',
                       opts: dict = None) -> WebSocketResponse:
        '''
        Executes a code snippet in the streaming mode.
        Since the returned websocket represents a run loop, there is no need to
        specify *run_id* explicitly.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        opts = {} if opts is None else opts
        if mode == 'query':
            opts = {}
        elif mode == 'batch':
            opts = {
                'clean': opts.get('clean', None),
                'build': opts.get('build', None),
                'buildLog': bool(opts.get('buildLog', False)),
                'exec': opts.get('exec', None),
            }
        else:
            msg = 'Invalid stream-execution mode: {0}'.format(mode)
            raise BackendClientError(msg)
        request = Request(self.session,
                          'GET', '/stream/kernel/{}/execute'.format(self.kernel_id),
                          params=params)

        async def send_code(ws):
            await ws.send_json({
                'code': code,
                'mode': mode,
                'options': opts,
            })

        return request.connect_websocket(on_enter=send_code)