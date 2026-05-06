def use_active_sequence_rule_enabler_view(self):
        """Pass through to provider SequenceRuleEnablerLookupSession.use_active_sequence_rule_enabler_view"""
        self._operable_views['sequence_rule_enabler'] = ACTIVE
        # self._get_provider_session('sequence_rule_enabler_lookup_session')  # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_active_sequence_rule_enabler_view()
            except AttributeError:
                pass