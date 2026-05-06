def get_repositories(self):
        """Pass through to provider RepositoryLookupSession.get_repositories"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('repository_lookup_session').get_repositories()
        cat_list = []
        for cat in catalogs:
            cat_list.append(Repository(self._provider_manager, cat, self._runtime, self._proxy))
        return RepositoryList(cat_list)