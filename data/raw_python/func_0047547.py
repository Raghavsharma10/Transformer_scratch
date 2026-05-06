def update_gradebook(self, *args, **kwargs):
        """Pass through to provider GradebookAdminSession.update_gradebook"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        # OSID spec does not require returning updated catalog
        return Gradebook(
            self._provider_manager,
            self._get_provider_session('gradebook_admin_session').update_gradebook(*args, **kwargs),
            self._runtime,
            self._proxy)