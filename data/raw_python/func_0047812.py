def get_objective_banks_by_activity(self, *args, **kwargs):
        """Pass through to provider ActivityObjectiveBankSession.get_objective_banks_by_activity"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceBinSession.get_bins_by_resource
        catalogs = self._get_provider_session('activity_objective_bank_session').get_objective_banks_by_activity(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(ObjectiveBank(self._provider_manager, cat, self._runtime, self._proxy))
        return ObjectiveBankList(cat_list)