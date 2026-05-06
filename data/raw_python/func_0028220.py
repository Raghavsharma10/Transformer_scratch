async def list_with_limit(cls,
                              limit,
                              offset,
                              status: str = 'ALIVE',
                              fields: Iterable[str] = None) -> Sequence[dict]:
        '''
        Fetches the list of agents with the given status with limit and offset for
        pagination.

        :param limit: number of agents to get
        :param offset: offset index of agents to get
        :param status: An upper-cased string constant representing agent
            status (one of ``'ALIVE'``, ``'TERMINATED'``, ``'LOST'``,
            etc.)
        :param fields: Additional per-agent query fields to fetch.
        '''
        if fields is None:
            fields = (
                'id',
                'addr',
                'status',
                'first_contact',
                'mem_slots',
                'cpu_slots',
                'gpu_slots',
            )
        q = 'query($limit: Int!, $offset: Int!, $status: String) {' \
            '  agent_list(limit: $limit, offset: $offset, status: $status) {' \
            '   items { $fields }' \
            '   total_count' \
            '  }' \
            '}'
        q = q.replace('$fields', ' '.join(fields))
        variables = {
            'limit': limit,
            'offset': offset,
            'status': status,
        }
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json({
            'query': q,
            'variables': variables,
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            return data['agent_list']