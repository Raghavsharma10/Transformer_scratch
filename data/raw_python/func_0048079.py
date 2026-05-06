def get_assessments_taken_by_query(self, assessment_taken_query):
        """Gets a list of ``AssessmentTaken`` elements matching the given assessment taken query.

        arg:    assessment_taken_query
                (osid.assessment.AssessmentTakenQuery): the assessment
                taken query
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTakenList``
        raise:  NullArgument - ``assessment_taken_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_taken_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in assessment_taken_query._query_terms:
            if '$in' in assessment_taken_query._query_terms[term] and '$nin' in assessment_taken_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': assessment_taken_query._query_terms[term]['$in']}},
                             {term: {'$nin': assessment_taken_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: assessment_taken_query._query_terms[term]})
        for term in assessment_taken_query._keyword_terms:
            or_list.append({term: assessment_taken_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('assessment',
                                             collection='AssessmentTaken',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.AssessmentTakenList(result, runtime=self._runtime, proxy=self._proxy)