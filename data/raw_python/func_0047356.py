def use_comparative_assessment_part_bank_view(self):
        """Pass through to provider AssessmentPartBankSession.use_comparative_assessment_part_bank_view"""
        self._bank_view = COMPARATIVE
        # self._get_provider_session('assessment_part_bank_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_bank_view()
            except AttributeError:
                pass