def get_assessments_offered_by_query(self, assessment_offered_query):
        """Gets a list of ``AssessmentOffered`` elements matching the given assessment offered query.

        arg:    assessment_offered_query
                (osid.assessment.AssessmentOfferedQuery): the assessment
                offered query
        return: (osid.assessment.AssessmentOfferedList) - the returned
                ``AssessmentOfferedList``
        raise:  NullArgument - ``assessment_offered_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_offered_query`` is not of
                this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in assessment_offered_query._query_terms:
            if '$in' in assessment_offered_query._query_terms[term] and '$nin' in assessment_offered_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': assessment_offered_query._query_terms[term]['$in']}},
                             {term: {'$nin': assessment_offered_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: assessment_offered_query._query_terms[term]})
        for term in assessment_offered_query._keyword_terms:
            or_list.append({term: assessment_offered_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('assessment',
                                             collection='AssessmentOffered',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.AssessmentOfferedList(result, runtime=self._runtime, proxy=self._proxy)