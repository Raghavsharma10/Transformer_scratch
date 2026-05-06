def get_activities_by_query(self, activity_query):
        """Gets a list of ``Activities`` matching the given activity query.

        arg:    activity_query (osid.learning.ActivityQuery): the
                activity query
        return: (osid.learning.ActivityList) - the returned
                ``ActivityList``
        raise:  NullArgument - ``activity_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``activity_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in activity_query._query_terms:
            if '$in' in activity_query._query_terms[term] and '$nin' in activity_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': activity_query._query_terms[term]['$in']}},
                             {term: {'$nin': activity_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: activity_query._query_terms[term]})
        for term in activity_query._keyword_terms:
            or_list.append({term: activity_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('learning',
                                             collection='Activity',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.ActivityList(result, runtime=self._runtime, proxy=self._proxy)