def get_grade_systems_by_query(self, grade_system_query):
        """Gets a list of ``GradeSystem`` objects matching the given grade system query.

        arg:    grade_system_query (osid.grading.GradeSystemQuery): the
                grade system query
        return: (osid.grading.GradeSystemList) - the returned
                ``GradeSystemList``
        raise:  NullArgument - ``grade_system_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_system_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in grade_system_query._query_terms:
            if '$in' in grade_system_query._query_terms[term] and '$nin' in grade_system_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': grade_system_query._query_terms[term]['$in']}},
                             {term: {'$nin': grade_system_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: grade_system_query._query_terms[term]})
        for term in grade_system_query._keyword_terms:
            or_list.append({term: grade_system_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('grading',
                                             collection='GradeSystem',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.GradeSystemList(result, runtime=self._runtime, proxy=self._proxy)