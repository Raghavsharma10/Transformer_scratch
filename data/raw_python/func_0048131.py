def use_federated_family_view(self):
        """Pass through to provider RelationshipLookupSession.use_federated_family_view"""
        self._family_view = FEDERATED
        # self._get_provider_session('relationship_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_federated_family_view()
            except AttributeError:
                pass