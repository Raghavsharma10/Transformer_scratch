def use_plenary_assessment_part_view(self):
        """Pass through to provider AssessmentPartLookupSession.use_plenary_assessment_part_view"""
        self._object_views['assessment_part'] = PLENARY
        # self._get_provider_session('assessment_part_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_assessment_part_view()
            except AttributeError:
                pass