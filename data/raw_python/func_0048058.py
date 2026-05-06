def get_assessments_offered_for_assessment(self, assessment_id):
        """Gets an ``AssessmentOfferedList`` by the given assessment.

        In plenary mode, the returned list contains all known
        assessments offered or an error results. Otherwise, the returned
        list may contain only those assessments offered that are
        accessible through this session.

        arg:    assessment_id (osid.id.Id): ``Id`` of an ``Assessment``
        return: (osid.assessment.AssessmentOfferedList) - the returned
                ``AssessmentOffered`` list
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityLookupSession.get_activities_for_objective_template
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentId': str(assessment_id)},
                 **self._view_filter()))
        return objects.AssessmentOfferedList(result, runtime=self._runtime)