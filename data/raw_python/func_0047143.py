def get_objectives_by_query(self, objective_query):
        """Gets a list of ``Objectives`` matching the given objective query.

        arg:    objective_query (osid.learning.ObjectiveQuery): the
                objective query
        return: (osid.learning.ObjectiveList) - the returned
                ``ObjectiveList``
        raise:  NullArgument - ``objective_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``objective_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in objective_query._query_terms:
            if '$in' in objective_query._query_terms[term] and '$nin' in objective_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': objective_query._query_terms[term]['$in']}},
                             {term: {'$nin': objective_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: objective_query._query_terms[term]})
        for term in objective_query._keyword_terms:
            or_list.append({term: objective_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('learning',
                                             collection='Objective',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.ObjectiveList(result, runtime=self._runtime, proxy=self._proxy)