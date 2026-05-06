async def create(cls, name: str,
                     default_for_unspecified: int,
                     total_resource_slots: int,
                     max_concurrent_sessions: int,
                     max_containers_per_session: int,
                     max_vfolder_count: int,
                     max_vfolder_size: int,
                     idle_timeout: int,
                     allowed_vfolder_hosts: Sequence[str],
                     fields: Iterable[str] = None) -> dict:
        """
        Creates a new keypair resource policy with the given options.
        You need an admin privilege for this operation.
        """
        if fields is None:
            fields = ('name',)
        q = 'mutation($name: String!, $input: CreateKeyPairResourcePolicyInput!) {' \
            + \
            '  create_keypair_resource_policy(name: $name, props: $input) {' \
            '    ok msg resource_policy { $fields }' \
            '  }' \
            '}'
        q = q.replace('$fields', ' '.join(fields))
        variables = {
            'name': name,
            'input': {
                'default_for_unspecified': default_for_unspecified,
                'total_resource_slots': total_resource_slots,
                'max_concurrent_sessions': max_concurrent_sessions,
                'max_containers_per_session': max_containers_per_session,
                'max_vfolder_count': max_vfolder_count,
                'max_vfolder_size': max_vfolder_size,
                'idle_timeout': idle_timeout,
                'allowed_vfolder_hosts': allowed_vfolder_hosts,
            },
        }
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
            'variables': variables,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['create_keypair_resource_policy']