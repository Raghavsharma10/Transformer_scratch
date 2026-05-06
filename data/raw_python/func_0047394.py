def save_sequence_rule(self, sequence_rule_form, *args, **kwargs):
        """Pass through to provider SequenceRuleAdminSession.update_sequence_rule"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if sequence_rule_form.is_for_update():
            return self.update_sequence_rule(sequence_rule_form, *args, **kwargs)
        else:
            return self.create_sequence_rule(sequence_rule_form, *args, **kwargs)