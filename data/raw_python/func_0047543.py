def use_plenary_gradebook_view(self):
        """Pass through to provider GradeSystemGradebookSession.use_plenary_gradebook_view"""
        self._gradebook_view = PLENARY
        # self._get_provider_session('grade_system_gradebook_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_gradebook_view()
            except AttributeError:
                pass