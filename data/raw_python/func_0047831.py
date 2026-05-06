def save_proficiency(self, proficiency_form, *args, **kwargs):
        """Pass through to provider ProficiencyAdminSession.update_proficiency"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if proficiency_form.is_for_update():
            return self.update_proficiency(proficiency_form, *args, **kwargs)
        else:
            return self.create_proficiency(proficiency_form, *args, **kwargs)