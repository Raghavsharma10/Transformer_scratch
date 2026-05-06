def save_activity(self, activity_form, *args, **kwargs):
        """Pass through to provider ActivityAdminSession.update_activity"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if activity_form.is_for_update():
            return self.update_activity(activity_form, *args, **kwargs)
        else:
            return self.create_activity(activity_form, *args, **kwargs)