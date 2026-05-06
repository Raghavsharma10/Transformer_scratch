def use_federated_bank_view(self):
        """Pass through to provider ItemLookupSession.use_federated_bank_view"""
        self._bank_view = FEDERATED
        # self._get_provider_session('item_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_federated_bank_view()
            except AttributeError:
                pass