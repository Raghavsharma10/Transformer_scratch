def _parse_parameters(self, resource, params):
        '''Creates a dictionary from query_string and `params`

        Transforms the `?key=value&...` to a {'key': 'value'} and adds
        (or overwrites if already present) the value with the dictionary in
        `params`.
        '''
        # remove params from resource URI (needed for paginated stuff)
        parsed_uri = urlparse(resource)
        qs = parsed_uri.query
        resource = urlunparse(parsed_uri._replace(query=''))
        prms = {}
        for tup in parse_qsl(qs):
            prms[tup[0]] = tup[1]

        # params supplied to self.get() override parsed params
        for key in params:
            prms[key] = params[key]
        return resource, prms