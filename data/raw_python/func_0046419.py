def get_assessment_taken(self):
        """Gets the ``AssessmentTakeb``.

        return: (osid.assessment.AssessmentTaken) - the assessment taken
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['assessmentTakenId']):
            raise errors.IllegalState('assessment_taken empty')
        mgr = self._get_provider_manager('ASSESSMENT')
        if not mgr.supports_assessment_taken_lookup():
            raise errors.OperationFailed('Assessment does not support AssessmentTaken lookup')
        lookup_session = mgr.get_assessment_taken_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        return lookup_session.get_assessment_taken(self.get_assessment_taken_id())