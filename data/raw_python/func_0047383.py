def use_any_status_assessment_part_view(self):
        """Pass through to provider AssessmentPartLookupSession.use_any_status_assessment_part_view"""
        self._operable_views['assessment_part'] = ANY_STATUS
        # self._get_provider_session('assessment_part_lookup_session')  # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_any_status_assessment_part_view()
            except AttributeError:
                pass