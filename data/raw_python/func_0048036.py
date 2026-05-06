def get_assessments_by_query(self, assessment_query):
        """Gets a list of ``Assessments`` matching the given assessment query.

        arg:    assessment_query (osid.assessment.AssessmentQuery): the
                assessment query
        return: (osid.assessment.AssessmentList) - the returned
                ``AssessmentList``
        raise:  NullArgument - ``assessment_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        """Gets a list of ``Assessments`` matching the given assessment query.

        arg:    assessment_query (osid.assessment.AssessmentQuery): the
                assessment query
        return: (osid.assessment.AssessmentList) - the returned
                ``AssessmentList``
        raise:  NullArgument - ``assessment_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_query`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        if 'assessmentOfferedId' in assessment_query._query_terms:
            collection = JSONClientValidated('assessment',
                                             collection='AssessmentOffered',
                                             runtime=self._runtime)
            match = '$in' in assessment_query._query_terms['assessmentOfferedId'].keys()
            if match:
                match_identifiers = [ObjectId(Id(i).identifier) for i in assessment_query._query_terms['assessmentOfferedId']['$in']]
                query = {'$in': match_identifiers}
            else:
                match_identifiers = [ObjectId(Id(i).identifier) for i in assessment_query._query_terms['assessmentOfferedId']['$in']]
                query = {'$nin': match_identifiers}

            result = collection.find({
                "_id": query
            })

            assessment_ids = [ObjectId(Id(r['assessmentId']).identifier) for r in result]

            collection = JSONClientValidated('assessment',
                                             collection='Assessment',
                                             runtime=self._runtime)
            result = collection.find({
                "_id": {"$in": assessment_ids}
            })
            return objects.AssessmentList(result, runtime=self._runtime, proxy=self._proxy)
        else:
            # and_list = list()
            # or_list = list()
            # for term in assessment_query._query_terms:
            #     and_list.append({term: assessment_query._query_terms[term]})
            # for term in assessment_query._keyword_terms:
            #     or_list.append({term: assessment_query._keyword_terms[term]})
            # if or_list:
            #     and_list.append({'$or': or_list})
            # view_filter = self._view_filter()
            # if view_filter:
            #     and_list.append(view_filter)
            # if and_list:
            #     query_terms = {'$and': and_list}
            #
            #     collection = JSONClientValidated('assessment',
            #                                      collection='Assessment',
            #                                      runtime=self._runtime)
            #     result = collection.find(query_terms).sort('_id', DESCENDING)
            # else:
            #     result = []
            # return objects.AssessmentList(result, runtime=self._runtime, proxy=self._proxy)
            and_list = list()
            or_list = list()
            for term in assessment_query._query_terms:
                if '$in' in assessment_query._query_terms[term] and '$nin' in assessment_query._query_terms[term]:
                    and_list.append(
                        {'$or': [{term: {'$in': assessment_query._query_terms[term]['$in']}},
                                 {term: {'$nin': assessment_query._query_terms[term]['$nin']}}]})
                else:
                    and_list.append({term: assessment_query._query_terms[term]})
            for term in assessment_query._keyword_terms:
                or_list.append({term: assessment_query._keyword_terms[term]})
            if or_list:
                and_list.append({'$or': or_list})
            view_filter = self._view_filter()
            if view_filter:
                and_list.append(view_filter)
            if and_list:
                query_terms = {'$and': and_list}
                collection = JSONClientValidated('assessment',
                                                 collection='Assessment',
                                                 runtime=self._runtime)
                result = collection.find(query_terms).sort('_id', DESCENDING)
            else:
                result = []
            return objects.AssessmentList(result, runtime=self._runtime, proxy=self._proxy)