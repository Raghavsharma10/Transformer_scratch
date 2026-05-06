def get_activities_by_query(self, activity_query=None):
        """Gets a list of Activities matching the given activity query.

        arg:    activityQuery (osid.learning.ActivityQuery): the
                activity query
        return: (osid.learning.ActivityList) - the returned ActivityList
        raise:  NullArgument - activityQuery is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - activityQuery is not of this service
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr)
        query_terms = [v for k, v in activity_query._query_terms.items()]
        url_path += '?' + '&'.join(query_terms)
        objects.ActivityList(self._get_request(url_path))