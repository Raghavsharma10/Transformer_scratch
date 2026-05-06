def get_users(self, filter=None, sortBy=None, sortOrder=None,
            startIndex=None, count=None):
        """
        Returns users accounts stored in UAA.
        See https://docs.cloudfoundry.org/api/uaa/#list63

        For filtering help, see:
        http://www.simplecloud.info/specs/draft-scim-api-01.html#query-resources
        """
        self.assert_has_permission('scim.read')

        params = {}
        if filter:
            params['filter'] = filter

        if sortBy:
            params['sortBy'] = sortBy

        if sortOrder:
            params['sortOrder'] = sortOrder

        if startIndex:
            params['startIndex'] = startIndex

        if count:
            params['count'] = count

        return self._get(self.uri + '/Users', params=params)