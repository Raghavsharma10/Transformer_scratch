def use_comparative_relationship_view(self):
        """Pass through to provider RelationshipLookupSession.use_comparative_relationship_view"""
        self._object_views['relationship'] = COMPARATIVE
        # self._get_provider_session('relationship_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_relationship_view()
            except AttributeError:
                pass