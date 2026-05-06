def get_rubric(self):
        """Gets the rubric.

        return: (osid.assessment.AssessmentTaken) - the assessment taken
        raise:  IllegalState - ``has_rubric()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['rubricId']):
            raise errors.IllegalState('this AssessmentTaken has no rubric')
        mgr = self._get_provider_manager('ASSESSMENT')
        if not mgr.supports_assessment_taken_lookup():
            raise errors.OperationFailed('Assessment does not support AssessmentTaken lookup')
        lookup_session = mgr.get_assessment_taken_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        osid_object = lookup_session.get_assessment_taken(self.get_rubric_id())
        return osid_object