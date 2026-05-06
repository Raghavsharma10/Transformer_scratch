async def execute(self, run_id: str = None,
                      code: str = None,
                      mode: str = 'query',
                      opts: dict = None):
        '''
        Executes a code snippet directly in the compute session or sends a set of
        build/clean/execute commands to the compute session.

        For more details about using this API, please refer :doc:`the official API
        documentation <user-api/intro>`.

        :param run_id: A unique identifier for a particular run loop.  In the
            first call, it may be ``None`` so that the server auto-assigns one.
            Subsequent calls must use the returned ``runId`` value to request
            continuation or to send user inputs.
        :param code: A code snippet as string.  In the continuation requests, it
            must be an empty string.  When sending user inputs, this is where the
            user input string is stored.
        :param mode: A constant string which is one of ``"query"``, ``"batch"``,
            ``"continue"``, and ``"user-input"``.
        :param opts: A dict for specifying additional options. Mainly used in the
            batch mode to specify build/clean/execution commands.
            See :ref:`the API object reference <batch-execution-query-object>`
            for details.

        :returns: :ref:`An execution result object <execution-result-object>`
        '''
        opts = opts if opts is not None else {}
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        if mode in {'query', 'continue', 'input'}:
            assert code is not None, \
                   'The code argument must be a valid string even when empty.'
            rqst = Request(self.session,
                'POST', '/kernel/{}'.format(self.kernel_id),
                params=params)
            rqst.set_json({
                'mode': mode,
                'code': code,
                'runId': run_id,
            })
        elif mode == 'batch':
            rqst = Request(self.session,
                'POST', '/kernel/{}'.format(self.kernel_id),
                params=params)
            rqst.set_json({
                'mode': mode,
                'code': code,
                'runId': run_id,
                'options': {
                    'clean': opts.get('clean', None),
                    'build': opts.get('build', None),
                    'buildLog': bool(opts.get('buildLog', False)),
                    'exec': opts.get('exec', None),
                },
            })
        elif mode == 'complete':
            rqst = Request(self.session,
                'POST', '/kernel/{}/complete'.format(self.kernel_id),
                params=params)
            rqst.set_json({
                'code': code,
                'options': {
                    'row': int(opts.get('row', 0)),
                    'col': int(opts.get('col', 0)),
                    'line': opts.get('line', ''),
                    'post': opts.get('post', ''),
                },
            })
        else:
            raise BackendClientError('Invalid execution mode: {0}'.format(mode))
        async with rqst.fetch() as resp:
            return (await resp.json())['result']