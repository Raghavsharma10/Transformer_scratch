def save_objective_bank(self, objective_bank_form, *args, **kwargs):
        """Pass through to provider ObjectiveBankAdminSession.update_objective_bank"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if objective_bank_form.is_for_update():
            return self.update_objective_bank(objective_bank_form, *args, **kwargs)
        else:
            return self.create_objective_bank(objective_bank_form, *args, **kwargs)