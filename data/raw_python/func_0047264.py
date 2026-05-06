def get_compositions_by_search(self, composition_query, composition_search):
        """Pass through to provider CompositionSearchSession.get_compositions_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_compositions_by_search(composition_query, composition_search)