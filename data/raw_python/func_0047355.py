def save_bank(self, bank_form, *args, **kwargs):
        """Pass through to provider BankAdminSession.update_bank"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if bank_form.is_for_update():
            return self.update_bank(bank_form, *args, **kwargs)
        else:
            return self.create_bank(bank_form, *args, **kwargs)