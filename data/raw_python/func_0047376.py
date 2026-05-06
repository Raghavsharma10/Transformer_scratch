def use_comparative_assessment_taken_view(self):
        """Pass through to provider AssessmentTakenLookupSession.use_comparative_assessment_taken_view"""
        self._object_views['assessment_taken'] = COMPARATIVE
        # self._get_provider_session('assessment_taken_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_assessment_taken_view()
            except AttributeError:
                pass