def get_grade_entries_by_query(self, grade_entry_query):
        """Gets a list of entries matching the given grade entry query.

        arg:    grade_entry_query (osid.grading.GradeEntryQuery): the
                grade entry query
        return: (osid.grading.GradeEntryList) - the returned
                ``GradeEntryList``
        raise:  NullArgument - ``grade_entry_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_entry_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in grade_entry_query._query_terms:
            if '$in' in grade_entry_query._query_terms[term] and '$nin' in grade_entry_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': grade_entry_query._query_terms[term]['$in']}},
                             {term: {'$nin': grade_entry_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: grade_entry_query._query_terms[term]})
        for term in grade_entry_query._keyword_terms:
            or_list.append({term: grade_entry_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('grading',
                                             collection='GradeEntry',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.GradeEntryList(result, runtime=self._runtime, proxy=self._proxy)