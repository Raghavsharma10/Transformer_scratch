def save_sequence_rule_enabler(self, sequence_rule_enabler_form, *args, **kwargs):
        """Pass through to provider SequenceRuleEnablerAdminSession.update_sequence_rule_enabler"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if sequence_rule_enabler_form.is_for_update():
            return self.update_sequence_rule_enabler(sequence_rule_enabler_form, *args, **kwargs)
        else:
            return self.create_sequence_rule_enabler(sequence_rule_enabler_form, *args, **kwargs)