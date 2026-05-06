def use_comparative_objective_view(self):
        """Pass through to provider ObjectiveLookupSession.use_comparative_objective_view"""
        self._object_views['objective'] = COMPARATIVE
        # self._get_provider_session('objective_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_objective_view()
            except AttributeError:
                pass