def get_bank(self, *args, **kwargs):
        """Pass through to provider BankLookupSession.get_bank"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bin
        return Bank(
            self._provider_manager,
            self._get_provider_session('bank_lookup_session').get_bank(*args, **kwargs),
            self._runtime,
            self._proxy)