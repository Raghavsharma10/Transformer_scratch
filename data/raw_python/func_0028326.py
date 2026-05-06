async def info(self, fields: Iterable[str] = None) -> dict:
        '''
        Returns the keypair's information such as resource limits.

        :param fields: Additional per-agent query fields to fetch.

        .. versionadded:: 18.12
        '''
        if fields is None:
            fields = (
                'access_key', 'secret_key',
                'is_active', 'is_admin',
            )
        q = 'query {' \
            '  keypair {' \
            '    $fields' \
            '  }' \
            '}'
        q = q.replace('$fields', ' '.join(fields))
        rqst = Request(self.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['keypair']