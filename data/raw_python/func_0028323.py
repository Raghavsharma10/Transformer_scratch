async def update(cls, access_key: str,
                     is_active: bool = None,
                     is_admin: bool = None,
                     resource_policy: str = None,
                     rate_limit: int = None) -> dict:
        """
        Creates a new keypair with the given options.
        You need an admin privilege for this operation.
        """
        q = 'mutation($access_key: String!, $input: ModifyKeyPairInput!) {' + \
            '  modify_keypair(access_key: $access_key, props: $input) {' \
            '    ok msg' \
            '  }' \
            '}'
        variables = {
            'access_key': access_key,
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
            return data['modify_keypair']