def use_plenary_assessment_part_item_view(self):
        """Pass through to provider AssessmentPartItemSession.use_plenary_assessment_part_item_view"""
        self._object_views['assessment_part_item'] = PLENARY
        # self._get_provider_session('assessment_part_item_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_assessment_part_item_view()
            except AttributeError:
                pass