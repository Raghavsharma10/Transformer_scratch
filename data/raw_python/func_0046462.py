def get_sequence_rule_enablers_by_search(self, sequence_rule_enabler_query, sequence_rule_enabler_search):
        """Pass through to provider SequenceRuleEnablerSearchSession.get_sequence_rule_enablers_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_sequence_rule_enablers_by_search(sequence_rule_enabler_query, sequence_rule_enabler_search)