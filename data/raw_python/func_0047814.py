def create_objective_bank(self, *args, **kwargs):
        """Pass through to provider ObjectiveBankAdminSession.create_objective_bank"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.create_bin
        return ObjectiveBank(
            self._provider_manager,
            self._get_provider_session('objective_bank_admin_session').create_objective_bank(*args, **kwargs),
            self._runtime,
            self._proxy)