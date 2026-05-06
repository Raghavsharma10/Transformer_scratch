def get_gradebooks_by_parent_genus_type(self, *args, **kwargs):
        """Pass through to provider GradebookLookupSession.get_gradebooks_by_parent_genus_type"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_by_parent_genus_type
        catalogs = self._get_provider_session('gradebook_lookup_session').get_gradebooks_by_parent_genus_type(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(Gradebook(self._provider_manager, cat, self._runtime, self._proxy))
        return GradebookList(cat_list)