def get_assessment_offered(self):
        """Gets the ``AssessmentOffered``.

        return: (osid.assessment.AssessmentOffered) - the assessment
                offered
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['assessmentOfferedId']):
            raise errors.IllegalState('assessment_offered empty')
        mgr = self._get_provider_manager('ASSESSMENT')
        if not mgr.supports_assessment_offered_lookup():
            raise errors.OperationFailed('Assessment does not support AssessmentOffered lookup')
        lookup_session = mgr.get_assessment_offered_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        return lookup_session.get_assessment_offered(self.get_assessment_offered_id())