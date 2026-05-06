def get_banks_by_item(self, *args, **kwargs):
        """Pass through to provider ItemBankSession.get_banks_by_item"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceBinSession.get_bins_by_resource
        catalogs = self._get_provider_session('item_bank_session').get_banks_by_item(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(Bank(self._provider_manager, cat, self._runtime, self._proxy))
        return BankList(cat_list)