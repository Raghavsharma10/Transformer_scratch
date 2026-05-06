def save_objective(self, objective_form, *args, **kwargs):
        """Pass through to provider ObjectiveAdminSession.update_objective"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if objective_form.is_for_update():
            return self.update_objective(objective_form, *args, **kwargs)
        else:
            return self.create_objective(objective_form, *args, **kwargs)