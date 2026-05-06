async def list(cls, user_id: Union[int, str] = None,
                   is_active: bool = None,
                   fields: Iterable[str] = None) -> Sequence[dict]:
        '''
        Lists the keypairs.
        You need an admin privilege for this operation.
        '''
        if fields is None:
            fields = (
                'access_key', 'secret_key',
                'is_active', 'is_admin',
            )
        if user_id is None:
            q = 'query($is_active: Boolean) {' \
                '  keypairs(is_active: $is_active) {' \
                '    $fields' \
                '  }' \
                '}'
        else:
            uid_type = 'Int!' if isinstance(user_id, int) else 'String!'
            q = 'query($user_id: {0}, $is_active: Boolean) {{'.format(uid_type) + \
                '  keypairs(user_id: $user_id, is_active: $is_active) {' \
                '    $fields' \
                '  }' \
                '}'
        q = q.replace('$fields', ' '.join(fields))
        variables = {
            'is_active': is_active,
        }
        if user_id is not None:
            variables['user_id'] = user_id
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
            'variables': variables,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['keypairs']