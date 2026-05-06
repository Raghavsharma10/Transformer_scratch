async def complete(self, code: str, opts: dict = None) -> Iterable[str]:
        '''
        Gets the auto-completion candidates from the given code string,
        as if a user has pressed the tab key just after the code in
        IDEs.

        Depending on the language of the compute session, this feature
        may not be supported.  Unsupported sessions returns an empty list.

        :param code: An (incomplete) code text.
        :param opts: Additional information about the current cursor position,
            such as row, col, line and the remainder text.

        :returns: An ordered list of strings.
        '''
        opts = {} if opts is None else opts
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
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
        async with rqst.fetch() as resp:
            return await resp.json()