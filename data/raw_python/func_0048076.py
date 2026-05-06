def get_assessments_taken_for_assessment_offered(self, assessment_offered_id):
        """Gets an ``AssessmentTakenList`` by the given assessment offered.

        In plenary mode, the returned list contains all known
        assessments taken or an error results. Otherwise, the returned
        list may contain only those assessments taken that are
        accessible through this session.

        arg:    assessment_offered_id (osid.id.Id): ``Id`` of an
                ``AssessmentOffered``
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTaken`` list
        raise:  NullArgument - ``assessment_offered_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityLookupSession.get_activities_for_objective_template
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentOfferedId': str(assessment_offered_id)},
                 **self._view_filter()))
        return objects.AssessmentTakenList(result, runtime=self._runtime)