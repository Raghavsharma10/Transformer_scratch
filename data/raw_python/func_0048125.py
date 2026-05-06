def update_family(self, *args, **kwargs):
        """Pass through to provider FamilyAdminSession.update_family"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        # OSID spec does not require returning updated catalog
        return Family(
            self._provider_manager,
            self._get_provider_session('family_admin_session').update_family(*args, **kwargs),
            self._runtime,
            self._proxy)