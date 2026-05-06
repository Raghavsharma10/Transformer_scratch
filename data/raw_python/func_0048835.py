def get_assessment_parts_by_query(self, assessment_part_query):
        """Gets a list of ``AssessmentParts`` matching the given assessment part query.

        arg:    assessment_part_query
                (osid.assessment.authoring.AssessmentPartQuery): the
                assessment part query
        return: (osid.assessment.authoring.AssessmentPartList) - the
                returned ``AssessmentPartList``
        raise:  NullArgument - ``assessment_part_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``assessment_part_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceQuerySession.get_resources_by_query
        and_list = list()
        or_list = list()
        for term in assessment_part_query._query_terms:
            if '$in' in assessment_part_query._query_terms[term] and '$nin' in assessment_part_query._query_terms[term]:
                and_list.append(
                    {'$or': [{term: {'$in': assessment_part_query._query_terms[term]['$in']}},
                             {term: {'$nin': assessment_part_query._query_terms[term]['$nin']}}]})
            else:
                and_list.append({term: assessment_part_query._query_terms[term]})
        for term in assessment_part_query._keyword_terms:
            or_list.append({term: assessment_part_query._keyword_terms[term]})
        if or_list:
            and_list.append({'$or': or_list})
        view_filter = self._view_filter()
        if view_filter:
            and_list.append(view_filter)
        if and_list:
            query_terms = {'$and': and_list}
            collection = JSONClientValidated('assessment_authoring',
                                             collection='AssessmentPart',
                                             runtime=self._runtime)
            result = collection.find(query_terms).sort('_id', DESCENDING)
        else:
            result = []
        return objects.AssessmentPartList(result, runtime=self._runtime, proxy=self._proxy)