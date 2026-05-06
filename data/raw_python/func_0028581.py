async def list(cls, fields: Iterable[str] = None) -> Sequence[dict]:
        '''
        Lists the keypair resource policies.
        You need an admin privilege for this operation.
        '''
        if fields is None:
            fields = (
                'name', 'created_at',
                'total_resource_slots', 'max_concurrent_sessions',
                'max_vfolder_count', 'max_vfolder_size',
                'idle_timeout',
            )
        q = 'query {' \
            '  keypair_resource_policies {' \
            '    $fields' \
            '  }' \
            '}'
        q = q.replace('$fields', ' '.join(fields))
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['keypair_resource_policies']