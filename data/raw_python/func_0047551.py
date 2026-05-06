def use_plenary_grade_system_view(self):
        """Pass through to provider GradeSystemLookupSession.use_plenary_grade_system_view"""
        self._object_views['grade_system'] = PLENARY
        # self._get_provider_session('grade_system_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_grade_system_view()
            except AttributeError:
                pass