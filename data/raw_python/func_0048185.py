def save_composition(self, composition_form, *args, **kwargs):
        """Pass through to provider CompositionAdminSession.update_composition"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if composition_form.is_for_update():
            return self.update_composition(composition_form, *args, **kwargs)
        else:
            return self.create_composition(composition_form, *args, **kwargs)