def use_active_sequence_rule_view(self):
        """Pass through to provider SequenceRuleLookupSession.use_active_sequence_rule_view"""
        self._operable_views['sequence_rule'] = ACTIVE
        # self._get_provider_session('sequence_rule_lookup_session')  # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_active_sequence_rule_view()
            except AttributeError:
                pass