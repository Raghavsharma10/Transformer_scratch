def create_hierarchy(self, *args, **kwargs):
        """Pass through to provider HierarchyAdminSession.create_hierarchy"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.create_bin
        return Hierarchy(
            self._provider_manager,
            self._get_provider_session('hierarchy_admin_session').create_hierarchy(*args, **kwargs),
            self._runtime,
            self._proxy)