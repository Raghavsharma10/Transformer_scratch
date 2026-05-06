def get_gradebook_columns_by_query(self, gradebook_column_query):
        """Gets a list of gradebook columns matching the given query.

        arg:    gradebook_column_query
                (osid.grading.GradebookColumnQuery): the gradebook
                column query
        return: (osid.grading.GradebookColumnList) - the returned
                ``GradebookColumnList``
        raise:  NullArgument - ``gradebook_column_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``gradebook_column_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in gradebook_column_query._query_terms:
            if '$in' in gradebook_column_query._query_terms[term] and '$nin' in gradebook_column_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': gradebook_column_query._query_terms[term]['$in']}},
                             {term: {'$nin': gradebook_column_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: gradebook_column_query._query_terms[term]})
        for term in gradebook_column_query._keyword_terms:
            or_list.append({term: gradebook_column_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('grading',
                                             collection='GradebookColumn',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.GradebookColumnList(result, runtime=self._runtime, proxy=self._proxy)