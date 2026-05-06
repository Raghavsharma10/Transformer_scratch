async def query(cls, query: str,
                    variables: Optional[Mapping[str, Any]] = None,
                    ) -> Any:
        '''
        Sends the GraphQL query and returns the response.

        :param query: The GraphQL query string.
        :param variables: An optional key-value dictionary
            to fill the interpolated template variables
            in the query.

        :returns: The object parsed from the response JSON string.
        '''
        gql_query = {
            'query': query,
            'variables': variables if variables else {},
        }
        rqst = Request(cls.session, 'POST', '/admin/graphql')
        rqst.set_json(gql_query)
        async with rqst.fetch() as resp:
            return await resp.json()