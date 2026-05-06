def use_plenary_sequence_rule_view(self):
        """Pass through to provider SequenceRuleLookupSession.use_plenary_sequence_rule_view"""
        self._object_views['sequence_rule'] = PLENARY
        # self._get_provider_session('sequence_rule_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_sequence_rule_view()
            except AttributeError:
                pass