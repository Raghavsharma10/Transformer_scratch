def get_banks(self):
        """Pass through to provider BankLookupSession.get_banks"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('bank_lookup_session').get_banks()
        cat_list = []
        for cat in catalogs:
            cat_list.append(Bank(self._provider_manager, cat, self._runtime, self._proxy))
        return BankList(cat_list)