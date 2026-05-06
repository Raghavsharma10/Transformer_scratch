def match(self, route):
        """
        Match input route and return new Message instance
        with parsed content
        """
        _resource = trim_resource(self.resource)
        self.method = self.method.lower()
        resource_match = route.resource_regex.search(_resource)
        if resource_match is None:
            return None

        # build params and querystring
        params = resource_match.groupdict()
        querystring = params.pop("querystring", "")
        setattr(self, "param", params)
        setattr(self, "query", parse_querystring(querystring))

        return copy.deepcopy(self)