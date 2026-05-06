def use_plenary_relationship_view(self):
        """Pass through to provider RelationshipLookupSession.use_plenary_relationship_view"""
        self._object_views['relationship'] = PLENARY
        # self._get_provider_session('relationship_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_relationship_view()
            except AttributeError:
                pass