def save_hierarchy(self, hierarchy_form, *args, **kwargs):
        """Pass through to provider HierarchyAdminSession.update_hierarchy"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if hierarchy_form.is_for_update():
            return self.update_hierarchy(hierarchy_form, *args, **kwargs)
        else:
            return self.create_hierarchy(hierarchy_form, *args, **kwargs)