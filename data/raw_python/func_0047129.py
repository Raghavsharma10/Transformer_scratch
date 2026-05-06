def create_book(self, *args, **kwargs):
        """Pass through to provider BookAdminSession.create_book"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.create_bin
        return Book(
            self._provider_manager,
            self._get_provider_session('book_admin_session').create_book(*args, **kwargs),
            self._runtime,
            self._proxy)