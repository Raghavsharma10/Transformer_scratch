def create_repository(self, *args, **kwargs):
        """Pass through to provider RepositoryAdminSession.create_repository"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.create_bin
        return Repository(
            self._provider_manager,
            self._get_provider_session('repository_admin_session').create_repository(*args, **kwargs),
            self._runtime,
            self._proxy)