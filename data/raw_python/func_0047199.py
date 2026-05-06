def get_proficiencies_by_query(self, proficiency_query):
        """Gets a list of ``Proficiencies`` matching the given proficiency query.

        arg:    proficiency_query (osid.learning.ProficiencyQuery): the
                proficiency query
        return: (osid.learning.ProficiencyList) - the returned
                ``ProficiencyList``
        raise:  NullArgument - ``proficiency_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``proficiency_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in proficiency_query._query_terms:
            if '$in' in proficiency_query._query_terms[term] and '$nin' in proficiency_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': proficiency_query._query_terms[term]['$in']}},
                             {term: {'$nin': proficiency_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: proficiency_query._query_terms[term]})
        for term in proficiency_query._keyword_terms:
            or_list.append({term: proficiency_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('learning',
                                             collection='Proficiency',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.ProficiencyList(result, runtime=self._runtime, proxy=self._proxy)