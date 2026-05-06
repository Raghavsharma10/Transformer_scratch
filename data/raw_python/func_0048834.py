def get_assessment_parts_for_assessment_part(self, assessment_part_id):
        """Gets an ``AssessmentPart`` for the given assessment part.

        arg:    assessment_part_id (osid.id.Id): an assessment part ``Id``
        return: (osid.assessment.authoring.AssessmentPartList) - the
                returned ``AssessmentPart`` list
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # NOT IN SPEC - Implemented from
        # osid.assessment_authoring.AssessmentPartLookupSession.additional_methods
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentPartId': str(assessment_part_id)},
                 **self._view_filter()))
        return objects.AssessmentPartList(result, runtime=self._runtime)