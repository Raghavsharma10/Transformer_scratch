def use_plenary_assessment_view(self):
        """Pass through to provider AssessmentLookupSession.use_plenary_assessment_view"""
        self._object_views['assessment'] = PLENARY
        # self._get_provider_session('assessment_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_assessment_view()
            except AttributeError:
                pass