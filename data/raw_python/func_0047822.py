def get_objective_form(self, *args, **kwargs):
        """Pass through to provider ObjectiveAdminSession.get_objective_form_for_update"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.get_resource_form_for_update
        # This method might be a bit sketchy. Time will tell.
        if isinstance(args[-1], list) or 'objective_record_types' in kwargs:
            return self.get_objective_form_for_create(*args, **kwargs)
        else:
            return self.get_objective_form_for_update(*args, **kwargs)