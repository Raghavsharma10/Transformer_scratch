def get_resources_by_query(self, resource_query):
        """Gets a list of ``Resources`` matching the given resource query.

        arg:    resource_query (osid.resource.ResourceQuery): the
                resource query
        return: (osid.resource.ResourceList) - the returned
                ``ResourceList``
        raise:  NullArgument - ``resource_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``resource_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in resource_query._query_terms:
            if '$in' in resource_query._query_terms[term] and '$nin' in resource_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': resource_query._query_terms[term]['$in']}},
                             {term: {'$nin': resource_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: resource_query._query_terms[term]})
        for term in resource_query._keyword_terms:
            or_list.append({term: resource_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('resource',
                                             collection='Resource',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.ResourceList(result, runtime=self._runtime, proxy=self._proxy)