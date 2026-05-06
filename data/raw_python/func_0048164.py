def get_repositories_by_composition(self, *args, **kwargs):
        """Pass through to provider CompositionRepositorySession.get_repositories_by_composition"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceBinSession.get_bins_by_resource
        catalogs = self._get_provider_session('composition_repository_session').get_repositories_by_composition(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(Repository(self._provider_manager, cat, self._runtime, self._proxy))
        return RepositoryList(cat_list)