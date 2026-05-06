def use_comparative_assessment_offered_view(self):
        """Pass through to provider AssessmentOfferedLookupSession.use_comparative_assessment_offered_view"""
        self._object_views['assessment_offered'] = COMPARATIVE
        # self._get_provider_session('assessment_offered_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_assessment_offered_view()
            except AttributeError:
                pass