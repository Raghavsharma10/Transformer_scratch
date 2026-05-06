def _get_assessment_taken(self, assessment_taken_id):
        """Helper method for getting an AssessmentTaken objects given an Id."""
        if assessment_taken_id not in self._assessments_taken:
            mgr = self._get_provider_manager('ASSESSMENT')
            lookup_session = mgr.get_assessment_taken_lookup_session(proxy=self._proxy)  # Should this be _for_bank?
            lookup_session.use_federated_bank_view()
            self._assessments_taken[assessment_taken_id] = (
                lookup_session.get_assessment_taken(assessment_taken_id))
        return self._assessments_taken[assessment_taken_id]