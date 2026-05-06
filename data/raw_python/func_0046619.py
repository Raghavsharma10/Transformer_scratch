def get_hierarchies(self):
        """Pass through to provider HierarchyLookupSession.get_hierarchies"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('hierarchy_lookup_session').get_hierarchies()
        cat_list = []
        for cat in catalogs:
            cat_list.append(Hierarchy(self._provider_manager, cat, self._runtime, self._proxy))
        return HierarchyList(cat_list)