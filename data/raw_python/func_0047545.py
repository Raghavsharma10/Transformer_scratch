def get_gradebooks(self):
        """Pass through to provider GradebookLookupSession.get_gradebooks"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('gradebook_lookup_session').get_gradebooks()
        cat_list = []
        for cat in catalogs:
            cat_list.append(Gradebook(self._provider_manager, cat, self._runtime, self._proxy))
        return GradebookList(cat_list)