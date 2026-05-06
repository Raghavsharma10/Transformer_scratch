def get_rubric(self):
        """Gets the rubric.

        return: (osid.assessment.AssessmentOffered) - the assessment
                offered
        raise:  IllegalState - ``has_rubric()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['rubricId']):
            raise errors.IllegalState('this AssessmentOffered has no rubric')
        mgr = self._get_provider_manager('ASSESSMENT')
        if not mgr.supports_assessment_offered_lookup():
            raise errors.OperationFailed('Assessment does not support AssessmentOffered lookup')
        lookup_session = mgr.get_assessment_offered_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        osid_object = lookup_session.get_assessment_offered(self.get_rubric_id())
        return osid_object