def save_item(self, item_form, *args, **kwargs):
        """Pass through to provider ItemAdminSession.update_item"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if item_form.is_for_update():
            return self.update_item(item_form, *args, **kwargs)
        else:
            return self.create_item(item_form, *args, **kwargs)