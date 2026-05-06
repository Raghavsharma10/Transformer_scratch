def use_comparative_hierarchy_view(self):
        """Pass through to provider HierarchyLookupSession.use_comparative_hierarchy_view"""
        self._hierarchy_view = COMPARATIVE
        # self._get_provider_session('hierarchy_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_hierarchy_view()
            except AttributeError:
                pass