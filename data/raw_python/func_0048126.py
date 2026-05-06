def save_family(self, family_form, *args, **kwargs):
        """Pass through to provider FamilyAdminSession.update_family"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if family_form.is_for_update():
            return self.update_family(family_form, *args, **kwargs)
        else:
            return self.create_family(family_form, *args, **kwargs)