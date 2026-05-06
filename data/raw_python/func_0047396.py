def use_plenary_sequence_rule_enabler_view(self):
        """Pass through to provider SequenceRuleEnablerLookupSession.use_plenary_sequence_rule_enabler_view"""
        self._object_views['sequence_rule_enabler'] = PLENARY
        # self._get_provider_session('sequence_rule_enabler_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_sequence_rule_enabler_view()
            except AttributeError:
                pass