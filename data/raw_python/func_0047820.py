def use_federated_objective_bank_view(self):
        """Pass through to provider ObjectiveLookupSession.use_federated_objective_bank_view"""
        self._objective_bank_view = FEDERATED
        # self._get_provider_session('objective_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_federated_objective_bank_view()
            except AttributeError:
                pass