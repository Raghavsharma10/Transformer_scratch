def save_asset(self, asset_form, *args, **kwargs):
        """Pass through to provider AssetAdminSession.update_asset"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if asset_form.is_for_update():
            return self.update_asset(asset_form, *args, **kwargs)
        else:
            return self.create_asset(asset_form, *args, **kwargs)