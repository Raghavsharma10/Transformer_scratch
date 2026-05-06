def get_objective_banks(self):
        """Pass through to provider ObjectiveBankLookupSession.get_objective_banks"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('objective_bank_lookup_session').get_objective_banks()
        cat_list = []
        for cat in catalogs:
            cat_list.append(ObjectiveBank(self._provider_manager, cat, self._runtime, self._proxy))
        return ObjectiveBankList(cat_list)