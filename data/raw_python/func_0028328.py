async def deactivate(cls, access_key: str) -> dict:
        '''
        Deactivates this keypair.
        Deactivated keypairs cannot make any API requests
        unless activated again by an administrator.
        You need an admin privilege for this operation.
        '''
        q = 'mutation($access_key: String!, $input: ModifyKeyPairInput!) {' + \
            '  modify_keypair(access_key: $access_key, props: $input) {' \
            '    ok msg' \
            '  }' \
            '}'
        variables = {
            'access_key': access_key,
            'input': {
                'is_active': False,
                'is_admin': None,
                'resource_policy': None,
                'rate_limit': None,
            },
        }
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
            'variables': variables,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['modify_keypair']