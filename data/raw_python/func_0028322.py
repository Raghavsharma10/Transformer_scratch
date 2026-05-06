async def create(cls, user_id: Union[int, str],
                     is_active: bool = True,
                     is_admin: bool = False,
                     resource_policy: str = None,
                     rate_limit: int = None,
                     fields: Iterable[str] = None) -> dict:
        '''
        Creates a new keypair with the given options.
        You need an admin privilege for this operation.
        '''
        if fields is None:
            fields = ('access_key', 'secret_key')
        uid_type = 'Int!' if isinstance(user_id, int) else 'String!'
        q = 'mutation($user_id: {0}, $input: KeyPairInput!) {{'.format(uid_type) + \
            '  create_keypair(user_id: $user_id, props: $input) {' \
            '    ok msg keypair { $fields }' \
            '  }' \
            '}'
        q = q.replace('$fields', ' '.join(fields))
        variables = {
            'user_id': user_id,
            'input': {
                'is_active': is_active,
                'is_admin': is_admin,
                'resource_policy': resource_policy,
                'rate_limit': rate_limit,
            },
        }
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
            'variables': variables,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['create_keypair']