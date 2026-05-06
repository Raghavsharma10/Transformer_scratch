def get_hierarchies_by_ids(self, *args, **kwargs):
        """Pass through to provider HierarchyLookupSession.get_hierarchies_by_ids"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_by_ids
        catalogs = self._get_provider_session('hierarchy_lookup_session').get_hierarchies_by_ids(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(Hierarchy(self._provider_manager, cat, self._runtime, self._proxy))
        return HierarchyList(cat_list)