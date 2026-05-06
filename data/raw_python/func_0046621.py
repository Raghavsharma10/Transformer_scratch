def get_hierarchy_form(self, *args, **kwargs):
        """Pass through to provider HierarchyAdminSession.get_hierarchy_form_for_update"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.get_bin_form_for_update_template
        # This method might be a bit sketchy. Time will tell.
        if isinstance(args[-1], list) or 'hierarchy_record_types' in kwargs:
            return self.get_hierarchy_form_for_create(*args, **kwargs)
        else:
            return self.get_hierarchy_form_for_update(*args, **kwargs)