def get_sequence_rules_for_assessment_part(self, assessment_part_id):
        """Gets a ``SequenceRuleList`` for the given source assessment part.

        arg:    assessment_part_id (osid.id.Id): an assessment part
                ``Id``
        return: (osid.assessment.authoring.SequenceRuleList) - the
                returned ``SequenceRule`` list
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityLookupSession.get_activities_for_objective_template
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentPartId': str(assessment_part_id)},
                 **self._view_filter()))
        return objects.SequenceRuleList(result, runtime=self._runtime)