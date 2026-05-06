def use_plenary_hierarchy_view(self):
        """Pass through to provider HierarchyLookupSession.use_plenary_hierarchy_view"""
        self._hierarchy_view = PLENARY
        # self._get_provider_session('hierarchy_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_hierarchy_view()
            except AttributeError:
                pass