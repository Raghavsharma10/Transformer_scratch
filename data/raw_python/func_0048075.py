def get_assessments_taken_for_assessment(self, assessment_id):
        """Gets an ``AssessmentTakenList`` for the given assessment.

        In plenary mode, the returned list contains all known
        assessments taken or an error results. Otherwise, the returned
        list may contain only those assessments taken that are
        accessible through this session.

        arg:    assessment_id (osid.id.Id): ``Id`` of an ``Assessment``
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTaken`` list
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentId': str(assessment_id)},
                 **self._view_filter())).sort('_id', DESCENDING)
        assessments_offered = objects.AssessmentOfferedList(
            result,
            runtime=self._runtime)

        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        ao_ids = []
        for assessment_offered in assessments_offered:
            ao_ids.append(str(assessment_offered.get_id()))

        result = collection.find(
            dict({'assessmentOfferedId': {'$in': ao_ids}},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentTakenList(result,
                                           runtime=self._runtime,
                                           proxy=self._proxy)