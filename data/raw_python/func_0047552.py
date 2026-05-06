def use_federated_gradebook_view(self):
        """Pass through to provider GradeSystemLookupSession.use_federated_gradebook_view"""
        self._gradebook_view = FEDERATED
        # self._get_provider_session('grade_system_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_federated_gradebook_view()
            except AttributeError:
                pass